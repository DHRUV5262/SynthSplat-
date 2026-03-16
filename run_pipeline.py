import os
import subprocess
import sys
import shutil

def check_cuda():
    print("Checking CUDA environment...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.version.cuda is None:
            print("\nCRITICAL ERROR: You have installed the CPU-only version of PyTorch.")
            print("gsplat requires a CUDA-enabled PyTorch build.")
            print("Please run the following command to install the correct version:")
            print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
            return False
            
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. gsplat requires a GPU with CUDA support.")
            return False
            
        # Check CUDA_HOME
        cuda_home = os.environ.get('CUDA_HOME')
        if not cuda_home:
            # Try to find it
            possible_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found CUDA at {path}. Setting CUDA_HOME.")
                    os.environ['CUDA_HOME'] = path
                    cuda_home = path
                    break
            
            # Fallback: try to find via nvcc in PATH
            if not cuda_home:
                nvcc_path = shutil.which('nvcc')
                if nvcc_path:
                    # nvcc is usually in bin/, so go up one level
                    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
                    print(f"Found CUDA via nvcc at {cuda_home}. Setting CUDA_HOME.")
                    os.environ['CUDA_HOME'] = cuda_home

            if not cuda_home:
                print("Warning: CUDA_HOME environment variable is not set. gsplat compilation may fail.")
                print("Please install CUDA Toolkit and set CUDA_HOME.")
        else:
            print(f"CUDA_HOME is set to: {cuda_home}")
            
        # Check for cl.exe (MSVC compiler)
        if not shutil.which('cl'):
            print("\nWarning: 'cl.exe' (MSVC compiler) not found in PATH.")
            print("gsplat JIT compilation requires Visual Studio C++ Build Tools.")
            print("Please run this script from a 'x64 Native Tools Command Prompt for VS 20xx'.")
            
        return True
    except ImportError:
        print("Error: torch is not installed.")
        return False

def main():
    # Paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    renderer_dir = os.path.join(root_dir, 'renderer')
    trainer_dir = os.path.join(root_dir, 'trainer')
    
    if not check_cuda():
        print("CUDA check failed. Proceeding anyway, but training may fail.")
    
    # 1. Run Renderer
    print("=== Step 1: Running Renderer ===")
    
    # Check for executable
    exe_path_release = os.path.join(renderer_dir, 'build', 'bin', 'Release', 'SynthSplat.exe')
    exe_path_debug = os.path.join(renderer_dir, 'build', 'bin', 'Debug', 'SynthSplat.exe')
    
    exe_path = None
    if os.path.exists(exe_path_release):
        exe_path = exe_path_release
    elif os.path.exists(exe_path_debug):
        exe_path = exe_path_debug
    else:
        print("Error: Renderer executable not found. Please build the renderer first.")
        sys.exit(1)
        
    print(f"Using renderer: {exe_path}")
    
    # Run renderer from renderer directory so it finds assets
    try:
        subprocess.check_call([exe_path], cwd=renderer_dir)
    except subprocess.CalledProcessError as e:
        print(f"Renderer failed with exit code {e.returncode}")
        sys.exit(1)
        
    # Verify output
    output_dir = os.path.join(renderer_dir, 'output')
    cameras_json = os.path.join(output_dir, 'cameras.json')
    if not os.path.exists(cameras_json):
        print("Error: Renderer did not produce output/cameras.json")
        sys.exit(1)
        
    print("Renderer finished successfully.")
    
    # 2. Run Training
    print("\n=== Step 2: Running Training ===")
    train_script = os.path.join(trainer_dir, 'train.py')
    
    try:
        subprocess.check_call([sys.executable, train_script], cwd=trainer_dir)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)
        
    print("Training finished successfully.")
    
    # 3. Run Evaluation
    print("\n=== Step 3: Running Evaluation ===")
    eval_script = os.path.join(trainer_dir, 'evaluate.py')
    
    try:
        subprocess.check_call([sys.executable, eval_script], cwd=trainer_dir)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        sys.exit(1)
        
    print("Evaluation finished successfully. Check trainer/eval_results/ for images.")

if __name__ == "__main__":
    main()
