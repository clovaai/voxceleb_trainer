#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
Knowledge Distillation Wrapper for Speaker Verification

Enables student models to learn from teacher models via:
1. Classification loss (AAM-Softmax on speaker labels)
2. Distillation loss (MSE/KL-divergence between embeddings)

Teacher: LSTM+Autoencoder (9.68% EER)
Student: MLP-Mixer (lightweight, fast)

Usage:
    - Set teacher_model and teacher_checkpoint in config
    - Set distillation_alpha (0.0-1.0): weight for distillation loss
    - Set distillation_temperature: softens teacher outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import os


class DistillationLoss(nn.Module):
    """
    Combined Classification + Distillation Loss
    
    Total Loss = (1-α) × Classification + α × Distillation
    
    Args:
        alpha: Weight for distillation loss (0.0 = no distillation, 1.0 = only distillation)
        temperature: Temperature for softening outputs (higher = softer)
        distillation_type: 'mse', 'cosine', or 'kl' (MSE/Cosine for embeddings, KL for logits)
    """
    def __init__(self, alpha=0.5, temperature=4.0, distillation_type='cosine'):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_type = distillation_type
        
        print(f'\nDistillation Loss Configuration:')
        print(f'  - Alpha: {alpha} (classification={1-alpha:.2f}, distillation={alpha:.2f})')
        print(f'  - Temperature: {temperature}')
        print(f'  - Type: {distillation_type}')
    
    def forward(self, student_embeddings, teacher_embeddings, classification_loss):
        """
        Compute combined loss
        
        Args:
            student_embeddings: Student model embeddings [batch, nOut]
            teacher_embeddings: Teacher model embeddings [batch, nOut]
            classification_loss: AAM-Softmax classification loss (scalar)
        
        Returns:
            total_loss: Combined loss
            distillation_loss: Distillation component (for logging)
        """
        # Distillation loss: Choose between MSE, Cosine, or KL
        if self.distillation_type == 'mse':
            # MSE between normalized embeddings (original implementation)
            student_norm = F.normalize(student_embeddings, p=2, dim=1)
            teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
            
            # MSE loss with temperature scaling
            distillation_loss = F.mse_loss(
                student_norm / self.temperature,
                teacher_norm / self.temperature
            )
            
        elif self.distillation_type == 'cosine':
            # Cosine similarity loss (RECOMMENDED - fixes magnitude issue)
            # Cosine similarity ranges from [-1, 1]
            # Loss = 1 - cosine_similarity ranges from [0, 2]
            cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1)
            distillation_loss = (1 - cos_sim).mean()
            
        elif self.distillation_type == 'kl':
            # KL divergence (if using logits instead of embeddings)
            student_soft = F.log_softmax(student_embeddings / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_embeddings / self.temperature, dim=1)
            distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            distillation_loss *= (self.temperature ** 2)  # Scale by T^2
        else:
            raise ValueError(f'Unknown distillation type: {self.distillation_type}')
        
        # Combined loss
        total_loss = (1 - self.alpha) * classification_loss + self.alpha * distillation_loss
        
        return total_loss, distillation_loss.item()


class TeacherModelWrapper(nn.Module):
    """
    Wrapper for teacher model (frozen, no gradients)
    
    Loads pre-trained teacher checkpoint and extracts embeddings
    """
    def __init__(self, model_name, checkpoint_path, **kwargs):
        super(TeacherModelWrapper, self).__init__()
        
        print(f'\nLoading Teacher Model: {model_name}')
        print(f'Checkpoint: {checkpoint_path}')
        
        # Load teacher model architecture
        TeacherModel = importlib.import_module("models." + model_name).__getattribute__("MainModel")
        self.model = TeacherModel(**kwargs)
        
        # Load pre-trained weights
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.__S__.'):
                    new_key = k.replace('module.__S__.', '')
                elif k.startswith('__S__.'):
                    new_key = k.replace('__S__.', '')
                elif k.startswith('module.'):
                    new_key = k.replace('module.', '')
                else:
                    new_key = k
                new_state_dict[new_key] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f'✓ Teacher weights loaded successfully')
        else:
            raise FileNotFoundError(f'Teacher checkpoint not found: {checkpoint_path}')
        
        # Freeze teacher parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Teacher parameters: {total_params:,}')
    
    def forward(self, x):
        """Extract teacher embeddings (no gradients)"""
        with torch.no_grad():
            return self.model(x)


class DistillationSpeakerNet(nn.Module):
    """
    Student model with knowledge distillation from teacher
    
    Combines:
    1. Student model (MLP-Mixer)
    2. Teacher model (LSTM+Autoencoder, frozen)
    3. Classification loss (AAM-Softmax)
    4. Distillation loss (MSE between embeddings)
    """
    def __init__(self, student_model, teacher_model, teacher_checkpoint,
                 optimizer, trainfunc, nPerSpeaker,
                 distillation_alpha=0.5, distillation_temperature=4.0,
                 distillation_type='cosine', freeze_teacher=True, **kwargs):
        super(DistillationSpeakerNet, self).__init__()
        
        print('\n' + '='*60)
        print('Initializing Knowledge Distillation Training')
        print('='*60)
        
        # Student model
        print(f'\nStudent Model: {student_model}')
        StudentModel = importlib.import_module("models." + student_model).__getattribute__("MainModel")
        self.__S__ = StudentModel(**kwargs)
        
        # Teacher model (frozen)
        if teacher_model and teacher_checkpoint:
            self.__T__ = TeacherModelWrapper(
                teacher_model, 
                teacher_checkpoint,
                **kwargs
            )
            self.use_distillation = True
        else:
            self.__T__ = None
            self.use_distillation = False
            print('\n⚠️  No teacher model specified - using classification only')
        
        # Classification loss (AAM-Softmax)
        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)
        
        # Distillation loss
        if self.use_distillation:
            self.__D__ = DistillationLoss(
                alpha=distillation_alpha,
                temperature=distillation_temperature,
                distillation_type=distillation_type
            )
        
        self.nPerSpeaker = nPerSpeaker
        
        # Print summary
        student_params = sum(p.numel() for p in self.__S__.parameters())
        print(f'\nStudent parameters: {student_params:,}')
        if self.use_distillation and self.__T__ is not None:
            teacher_params = sum(p.numel() for p in self.__T__.parameters())
            compression_ratio = teacher_params / student_params
            print(f'Compression ratio: {compression_ratio:.2f}× (teacher/student)')
        print('='*60 + '\n')
    
    def forward(self, data, label=None):
        """
        Forward pass with optional distillation
        
        Args:
            data: Input audio [batch, samples]
            label: Speaker labels (optional, for training)
        
        Returns:
            - Inference: Student embeddings
            - Training: (loss, accuracy, distillation_loss)
        """
        # Ensure correct shape
        if data.dim() == 3:
            data = data.reshape(-1, data.size()[-1])
        
        data = data.cuda(non_blocking=True)
        
        # Student forward pass
        student_output = self.__S__.forward(data)
        
        # Inference mode
        if label is None:
            return student_output
        
        # Training mode
        # Reshape for classification loss
        student_output_reshaped = student_output.reshape(
            self.nPerSpeaker, -1, student_output.size()[-1]
        ).transpose(1, 0).squeeze(1)
        
        # Classification loss
        classification_loss, prec1 = self.__L__.forward(student_output_reshaped, label)
        
        # Distillation loss (if teacher exists)
        if self.use_distillation and self.training:
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_output = self.__T__.forward(data)
            
            # Reshape teacher output to match student
            teacher_output_reshaped = teacher_output.reshape(
                self.nPerSpeaker, -1, teacher_output.size()[-1]
            ).transpose(1, 0).squeeze(1)
            
            # Combined loss
            total_loss, distillation_loss_value = self.__D__.forward(
                student_output_reshaped,
                teacher_output_reshaped,
                classification_loss
            )
            
            return total_loss, prec1, distillation_loss_value
        else:
            # No distillation - return classification loss only
            return classification_loss, prec1, 0.0


def create_distillation_model(config):
    """
    Factory function to create distillation-enabled model
    
    Args:
        config: Configuration dict with distillation parameters
    
    Returns:
        model: DistillationSpeakerNet instance
    """
    teacher_model = config.get('teacher_model', None)
    teacher_checkpoint = config.get('teacher_checkpoint', None)
    
    # If no teacher specified, use standard SpeakerNet
    if not teacher_model or not teacher_checkpoint:
        from SpeakerNet_performance_updated import SpeakerNet
        return SpeakerNet(**config)
    
    # Create distillation model
    return DistillationSpeakerNet(
        student_model=config['model'],
        teacher_model=teacher_model,
        teacher_checkpoint=teacher_checkpoint,
        distillation_alpha=config.get('distillation_alpha', 0.5),
        distillation_temperature=config.get('distillation_temperature', 4.0),
        freeze_teacher=config.get('freeze_teacher', True),
        **config
    )
