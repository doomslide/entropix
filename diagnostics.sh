#!/bin/bash

# ===================================================================
# Script Name: generate_comprehensive_setup_report.sh
# Description: Gathers system diagnostics and identifies relevant
#              directories for Python, CUDA, JAX, XLA, and NVIDIA setups.
# Author: Your Name
# Date: 2024-04-27
# ===================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ------------------------------
# Configuration
# ------------------------------

# Define the output report file
REPORT_FILE="comprehensive_setup_report_$(date +%F_%T).txt"

# Define directories to exclude from search (customize as needed)
EXCLUDE_PATHS=(
  "/home/$USER/.local/share/Trash/*"
  "/home/$USER/.cache/pypoetry/virtualenvs/*"
  "/home/$USER/Projects/*/.venv/*"
  "/home/$USER/.cache/pip/*"
  "/home/$USER/.local/lib/*"
  "/snap/*"
  "/var/lib/*"
  "/usr/share/*"
  "/proc/*"
  "/sys/*"
)

# Primary CUDA directory to keep
PRIMARY_CUDA_DIR="/usr/local/cuda"

# ------------------------------
# Functions
# ------------------------------

# Function to display and log messages
log() {
  echo -e "$1" | tee -a "$REPORT_FILE"
}

# Function to gather system information
gather_system_info() {
  log "\n===== System Information =====\n"

  # OS Information
  log "Operating System:"
  lsb_release -a 2>/dev/null | tee -a "$REPORT_FILE"
  
  # Kernel Information
  log "\nKernel Version:"
  uname -r | tee -a "$REPORT_FILE"

  # CPU Information
  log "\n===== CPU Information =====\n"
  lscpu | grep -E "Architecture|CPU\(s\)|Thread|Core|Socket|MHz" | tee -a "$REPORT_FILE"

  # Memory Information
  log "\n===== Memory Information =====\n"
  free -h | tee -a "$REPORT_FILE"
  log "\nSwap Configuration:"
  swapon --show 2>/dev/null | tee -a "$REPORT_FILE"
}

# Function to gather NVIDIA GPU and CUDA information
gather_gpu_cuda_info() {
  log "\n===== NVIDIA GPU Information =====\n"
  if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | tee -a "$REPORT_FILE"
    log "\n✓ nvidia-smi available\n"
    
    log "GPU Compute Mode:"
    nvidia-smi --query-gpu=compute_mode --format=csv | tee -a "$REPORT_FILE"
    
    log "\nGPU Power State:"
    nvidia-smi --query-gpu=pstate --format=csv | tee -a "$REPORT_FILE"
    
    log "\nGPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$REPORT_FILE"
  else
    log "\n✗ nvidia-smi not found\n"
  fi

  log "\n===== CUDA Configuration =====\n"
  log "CUDA Path:"
  ls -l /usr/local/cuda 2>/dev/null | tee -a "$REPORT_FILE" || log "CUDA path not found.\n"

  log "\nCUDA Version:"
  nvcc --version 2>/dev/null | tee -a "$REPORT_FILE" || log "nvcc not found.\n"

  log "\nCUDA Libraries:"
  ldconfig -p | grep cuda | tee -a "$REPORT_FILE" || log "No CUDA libraries found.\n"

  log "\nCUDA Environment Variables:"
  env | grep -i cuda | tee -a "$REPORT_FILE" || log "No CUDA environment variables set.\n"
}

# Function to gather cuDNN information
gather_cudnn_info() {
  log "\n===== cuDNN Configuration =====\n"
  if [ -f /usr/include/cudnn.h ]; then
    log "cuDNN Header:"
    ls -l /usr/include/cudnn.h | tee -a "$REPORT_FILE"
    
    log "cuDNN Version:"
    grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h 2>/dev/null | tee -a "$REPORT_FILE" || log "cudnn_version.h not found.\n"
  else
    log "\n✗ cuDNN headers not found in /usr/include\n"
  fi
}

# Function to gather Python environment information
gather_python_info() {
  log "\n===== Python Environment =====\n"
  
  log "Python Version:"
  python3 --version 2>&1 | tee -a "$REPORT_FILE"

  log "\nPython Path:"
  which python3 | tee -a "$REPORT_FILE"

  log "\nPIP Version:"
  pip3 --version 2>&1 | tee -a "$REPORT_FILE"

  log "\nInstalled JAX-related packages:"
  pip3 list | grep -iE "jax|cuda|nvidia" | tee -a "$REPORT_FILE" || log "No JAX-related packages found.\n"
}

# Function to gather XLA configuration
gather_xla_info() {
  log "\n===== XLA Configuration =====\n"
  
  log "XLA Environment Variables:"
  env | grep -i xla | tee -a "$REPORT_FILE" || log "No XLA environment variables set.\n"
}

# Function to gather library paths
gather_library_paths() {
  log "\n===== Library Paths =====\n"
  
  log "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | tee -a "$REPORT_FILE"
  log "LIBRARY_PATH: $LIBRARY_PATH" | tee -a "$REPORT_FILE"
  log "PATH: $PATH" | tee -a "$REPORT_FILE"
}

# Function to perform system checks
perform_system_checks() {
  log "\n===== System Checks =====\n"
  
  log "Loaded NVIDIA Kernel Modules:"
  lsmod | grep nvidia | tee -a "$REPORT_FILE" || log "No NVIDIA kernel modules loaded.\n"

  log "\nGPU Device Nodes:"
  ls -l /dev/nvidia* 2>/dev/null | tee -a "$REPORT_FILE" || log "No GPU device nodes found.\n"

  log "\nRecent GPU/CUDA related system messages:"
  journalctl -b | grep -i "nvidia\|cuda" | tail -n 10 | tee -a "$REPORT_FILE" || log "No recent GPU/CUDA related system messages found.\n"
}

# Function to perform a quick JAX test
perform_jax_test() {
  log "\n===== JAX Quick Test =====\n"
  
  log "Running minimal JAX test...\n" | tee -a "$REPORT_FILE"
  
  JAX_TEST_OUTPUT=$(python3 -c "
try:
    import jax
    import jax.numpy as jnp
    print(f'JAX version: {jax.__version__}')
    print(f'Available devices: {jax.devices()}')
    x = jnp.array([[1., 2.], [3., 4.]])
    y = jnp.array([[5., 6.], [7., 8.]])
    result = jnp.dot(x, y).block_until_ready()
    print('Basic matrix multiplication test successful')
except Exception as e:
    print(f'Error: {str(e)}')
")

  echo "$JAX_TEST_OUTPUT" | tee -a "$REPORT_FILE"
}

# Function to find relevant setup directories
find_relevant_directories() {
  log "\n===== Relevant Setup Directories =====\n"

  # Define exclusion parameters for find
  EXCLUDE_PARAMS=()
  for path in "${EXCLUDE_PATHS[@]}"; do
    EXCLUDE_PARAMS+=(-path "$path" -prune -o)
  done

  # Find relevant directories and files
  log "Searching for CUDA, JAX, XLA, and NVIDIA related directories and files...\n" | tee -a "$REPORT_FILE"
  
  # CUDA Directories
  CUDA_DIRS=$(find / -type d \( "${EXCLUDE_PARAMS[@]}" \) -name "cuda*" 2>/dev/null)
  if [ -n "$CUDA_DIRS" ]; then
    log "CUDA Installation Directories:" | tee -a "$REPORT_FILE"
    echo "$CUDA_DIRS" | tee -a "$REPORT_FILE"
  else
    log "No CUDA installation directories found." | tee -a "$REPORT_FILE"
  fi

  # JAX Directories
  JAX_DIRS=$(find / -type d \( "${EXCLUDE_PARAMS[@]}" \) -name "jax*" 2>/dev/null)
  if [ -n "$JAX_DIRS" ]; then
    log "\nJAX Installation Directories:" | tee -a "$REPORT_FILE"
    echo "$JAX_DIRS" | tee -a "$REPORT_FILE"
  else
    log "\nNo JAX installation directories found." | tee -a "$REPORT_FILE"
  fi

  # XLA Directories
  XLA_DIRS=$(find / -type d \( "${EXCLUDE_PARAMS[@]}" \) -name "xla*" 2>/dev/null)
  if [ -n "$XLA_DIRS" ]; then
    log "\nXLA Component Directories:" | tee -a "$REPORT_FILE"
    echo "$XLA_DIRS" | tee -a "$REPORT_FILE"
  else
    log "\nNo XLA component directories found." | tee -a "$REPORT_FILE"
  fi

  # NVIDIA Libraries
  NVIDIA_LIBS=$(find / -type f \( "${EXCLUDE_PARAMS[@]}" \) \( -name "libcuda.so*" -o -name "libcudart.so*" -o -name "libcublas.so*" -o -name "libcudnn.so*" \) 2>/dev/null)
  if [ -n "$NVIDIA_LIBS" ]; then
    log "\nNVIDIA Libraries:" | tee -a "$REPORT_FILE"
    echo "$NVIDIA_LIBS" | tee -a "$REPORT_FILE"
  else
    log "\nNo NVIDIA libraries found." | tee -a "$REPORT_FILE"
  fi

  # Python CUDA Bindings
  PYCUDA_BINDINGS=$(find / -type d \( "${EXCLUDE_PARAMS[@]}" \) \( -name "pycuda" -o -name "cupy" -o -name "jax" \) 2>/dev/null)
  if [ -n "$PYCUDA_BINDINGS" ]; then
    log "\nPython CUDA Bindings:" | tee -a "$REPORT_FILE"
    echo "$PYCUDA_BINDINGS" | tee -a "$REPORT_FILE"
  else
    log "\nNo Python CUDA bindings found." | tee -a "$REPORT_FILE"
  fi

  # Environment Configuration Files
  log "\n===== Environment Configuration Files =====\n" | tee -a "$REPORT_FILE"
  
  CONFIG_FILES=(
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.zshrc"
    "$HOME/.profile"
  )

  for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
      MATCHES=$(grep -E "CUDA|JAX|XLA" "$config" || true)
      if [ -n "$MATCHES" ]; then
        log "Configuration in $config:" | tee -a "$REPORT_FILE"
        echo "$MATCHES" | tee -a "$REPORT_FILE"
      fi
    fi
  done
}

# Function to summarize the report
summarize_report() {
  log "\n===== Report Summary =====\n"
  log "The comprehensive setup report has been saved to '$REPORT_FILE'."
  log "Please review the report for detailed information about your system's Python, CUDA, JAX, XLA, and NVIDIA configurations."
}

# ------------------------------
# Main Script Execution
# ------------------------------

# Initialize the report file
echo "===== Comprehensive Setup Report =====" > "$REPORT_FILE"
echo "Report Generated on: $(date)" >> "$REPORT_FILE"

# Gather system information
gather_system_info

# Gather GPU and CUDA information
gather_gpu_cuda_info

# Gather cuDNN information
gather_cudnn_info

# Gather Python environment information
gather_python_info

# Gather XLA configuration
gather_xla_info

# Gather library paths
gather_library_paths

# Perform system checks
perform_system_checks

# Perform JAX test
perform_jax_test

# Find relevant setup directories
find_relevant_directories

# Summarize the report
summarize_report