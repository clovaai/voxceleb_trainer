#!/usr/bin/env python3
"""
Comprehensive repository debugging script for voxceleb_trainer
This script checks:
1. Syntax errors
2. Import issues
3. Configuration files
4. Data paths
5. Dependencies
6. Code quality issues
"""

import os
import sys
import importlib
import yaml
import subprocess
from pathlib import Path

class RepoDebugger:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def print_section(self, title):
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
    
    def check_syntax(self):
        """Check Python syntax in all .py files"""
        self.print_section("1. CHECKING PYTHON SYNTAX")
        
        py_files = list(self.repo_path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files")
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
                # print(f"  ✓ {py_file.relative_to(self.repo_path)}")
            except SyntaxError as e:
                error_msg = f"Syntax error in {py_file.relative_to(self.repo_path)}: {e}"
                self.errors.append(error_msg)
                print(f"  ❌ {error_msg}")
        
        if not self.errors:
            print("✓ No syntax errors found")
            self.passed.append("Syntax check")
    
    def check_imports(self):
        """Check if all imports are available"""
        self.print_section("2. CHECKING IMPORTS")
        
        # Main modules to check
        modules = ['trainSpeakerNet.py', 'SpeakerNet.py', 'DatasetLoader.py']
        
        for module_name in modules:
            module_path = self.repo_path / module_name
            if not module_path.exists():
                self.warnings.append(f"Module not found: {module_name}")
                continue
            
            print(f"\nChecking imports in {module_name}...")
            try:
                # Read file and extract imports
                with open(module_path, 'r') as f:
                    content = f.read()
                
                # Parse imports
                import ast
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._check_module(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self._check_module(node.module)
                
                print(f"  ✓ All imports available in {module_name}")
            except Exception as e:
                error_msg = f"Error checking {module_name}: {e}"
                self.errors.append(error_msg)
                print(f"  ❌ {error_msg}")
    
    def _check_module(self, module_name):
        """Helper to check if a module is importable"""
        try:
            # Skip relative imports and standard library
            if module_name.startswith('.'):
                return
            
            base_module = module_name.split('.')[0]
            importlib.import_module(base_module)
        except ImportError as e:
            if base_module not in ['models', 'loss', 'DatasetLoader']:  # Skip local modules
                warning_msg = f"  ⚠ Import warning: {module_name} - {e}"
                if warning_msg not in self.warnings:
                    self.warnings.append(warning_msg)
    
    def check_config_files(self):
        """Check YAML configuration files"""
        self.print_section("3. CHECKING CONFIGURATION FILES")
        
        config_files = list(self.repo_path.rglob("*.yaml"))
        print(f"Found {len(config_files)} YAML files")
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"  ✓ {config_file.relative_to(self.repo_path)}")
                
                # Check required fields
                if config_file.name == 'experiment_01.yaml':
                    self._validate_experiment_config(config, config_file)
                    
            except yaml.YAMLError as e:
                error_msg = f"YAML error in {config_file.relative_to(self.repo_path)}: {e}"
                self.errors.append(error_msg)
                print(f"  ❌ {error_msg}")
    
    def _validate_experiment_config(self, config, config_file):
        """Validate experiment configuration"""
        required_fields = ['train_list', 'test_list', 'train_path', 'test_path']
        
        for field in required_fields:
            if field not in config:
                warning_msg = f"Missing required field '{field}' in {config_file.name}"
                self.warnings.append(warning_msg)
                print(f"    ⚠ {warning_msg}")
            else:
                # Check if paths exist
                path = config[field]
                if isinstance(path, str) and path.startswith('/'):
                    if not os.path.exists(path):
                        warning_msg = f"Path does not exist: {field} = {path}"
                        self.warnings.append(warning_msg)
                        print(f"    ⚠ {warning_msg}")
                    else:
                        print(f"    ✓ {field}: {path}")
    
    def check_data_paths(self):
        """Check if data paths exist"""
        self.print_section("4. CHECKING DATA PATHS")
        
        config_file = self.repo_path / 'configs' / 'experiment_01.yaml'
        if not config_file.exists():
            print("  ⚠ No experiment_01.yaml found")
            return
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        paths_to_check = {
            'train_list': config.get('train_list'),
            'test_list': config.get('test_list'),
            'train_path': config.get('train_path'),
            'test_path': config.get('test_path'),
            'musan_path': config.get('musan_path'),
            'rir_path': config.get('rir_path'),
        }
        
        for name, path in paths_to_check.items():
            if path and os.path.exists(path):
                print(f"  ✓ {name}: {path}")
            elif path:
                error_msg = f"Path not found: {name} = {path}"
                self.errors.append(error_msg)
                print(f"  ❌ {error_msg}")
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        self.print_section("5. CHECKING DEPENDENCIES")
        
        required_packages = [
            'torch', 'torchaudio', 'numpy', 'scipy', 'pandas',
            'yaml', 'tensorboard', 'sklearn', 'matplotlib'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                print(f"  ✓ {package}")
            except ImportError:
                error_msg = f"Missing package: {package}"
                self.errors.append(error_msg)
                print(f"  ❌ {error_msg}")
    
    def check_code_quality(self):
        """Run basic code quality checks"""
        self.print_section("6. CODE QUALITY CHECKS")
        
        # Check for common issues
        py_files = list(self.repo_path.rglob("*.py"))
        
        issues = {
            'print_statements': 0,
            'pdb_breakpoints': 0,
            'long_lines': 0,
        }
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if 'import pdb' in line or 'pdb.set_trace()' in line:
                            issues['pdb_breakpoints'] += 1
                            print(f"  ⚠ Debug breakpoint in {py_file.name}:{line_num}")
                        
                        if len(line) > 120:
                            issues['long_lines'] += 1
            except Exception:
                pass
        
        print(f"\nSummary:")
        print(f"  Debug breakpoints found: {issues['pdb_breakpoints']}")
        print(f"  Lines > 120 chars: {issues['long_lines']}")
    
    def print_summary(self):
        """Print final summary"""
        self.print_section("DEBUGGING SUMMARY")
        
        print(f"\n✅ Passed checks: {len(self.passed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.warnings:
            print("\n--- WARNINGS ---")
            for warning in self.warnings[:10]:
                print(f"  ⚠ {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        if self.errors:
            print("\n--- ERRORS ---")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        if not self.errors:
            print("\n✅ Repository is ready for use!")
            return 0
        else:
            print(f"\n❌ Found {len(self.errors)} errors. Please fix them before proceeding.")
            return 1

def main():
    repo_path = '/mnt/ricproject3/2025/Colvaiai/voxceleb_trainer'
    
    print("=" * 80)
    print("VOXCELEB TRAINER REPOSITORY DEBUGGER")
    print("=" * 80)
    print(f"Repository: {repo_path}\n")
    
    debugger = RepoDebugger(repo_path)
    
    # Run all checks
    debugger.check_syntax()
    debugger.check_imports()
    debugger.check_config_files()
    debugger.check_data_paths()
    debugger.check_dependencies()
    debugger.check_code_quality()
    
    # Print summary
    return debugger.print_summary()

if __name__ == "__main__":
    sys.exit(main())
