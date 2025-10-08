#!/bin/bash
# Unified Experiment Runner for TGVAnn
# Supports both Maize (TMCI) and Sugarcane (KSCI) datasets

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    echo "TGVAnn Experiment Runner"
    echo ""
    echo "Usage: $0 [DATASET] [ACTION] [OPTIONS]"
    echo ""
    echo "DATASETS:"
    echo "  maize      - TMCI Maize dataset (3 classes)"
    echo "  sugarcane  - KSCI Sugarcane dataset (5 classes)"
    echo ""
    echo "ACTIONS:"
    echo "  texture    - Generate LDP texture features"
    echo "  train      - Train the model"
    echo "  eval       - Evaluate trained model"
    echo "  gradcam    - Generate Grad-CAM visualizations"
    echo "  all        - Run complete pipeline"
    echo ""
    echo "OPTIONS:"
    echo "  --quick    - Quick run with reduced epochs (for testing)"
    echo "  --full     - Full production run"
    echo ""
    echo "Examples:"
    echo "  $0 maize all --quick"
    echo "  $0 sugarcane train --full"
    echo "  $0 maize gradcam"
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

DATASET=$1
ACTION=$2
MODE=${3:-"--full"}

# Validate dataset
if [ "$DATASET" != "maize" ] && [ "$DATASET" != "sugarcane" ]; then
    print_error "Invalid dataset: $DATASET"
    usage
fi

# Validate action
if [ "$ACTION" != "texture" ] && [ "$ACTION" != "train" ] && [ "$ACTION" != "eval" ] && [ "$ACTION" != "gradcam" ] && [ "$ACTION" != "all" ]; then
    print_error "Invalid action: $ACTION"
    usage
fi

# Set dataset-specific parameters
if [ "$DATASET" == "maize" ]; then
    RGB_DIR="./Maize_RGB"
    TEXTURE_DIR="./Maize_Texture"
    OUTPUT_DIR="./outputs/maize"
    BATCH_SIZE=32
    EPOCHS_FULL=50
    EPOCHS_QUICK=5
    LR=1e-4
    NUM_CLASSES=3
    TRAIN_SCRIPT="train.py"
    EVAL_SCRIPT="evaluate.py"
    GRADCAM_SCRIPT="visualize_gradcam.py"
    TEXTURE_SCRIPT="data/ldp_texture_generator.py"
else
    RGB_DIR="./Sugarcane_RGB"
    TEXTURE_DIR="./Sugarcane_Texture"
    OUTPUT_DIR="./outputs/sugarcane"
    BATCH_SIZE=16
    EPOCHS_FULL=100
    EPOCHS_QUICK=10
    LR=1e-5
    NUM_CLASSES=5
    TRAIN_SCRIPT="train_sugarcane.py"
    EVAL_SCRIPT="evaluate_sugarcane.py"
    GRADCAM_SCRIPT="visualize_gradcam_sugarcane.py"
    TEXTURE_SCRIPT="generate_texture_sugarcane.py"
fi

# Set epochs based on mode
if [ "$MODE" == "--quick" ]; then
    EPOCHS=$EPOCHS_QUICK
    print_warning "Quick mode: Using $EPOCHS epochs (for testing only)"
else
    EPOCHS=$EPOCHS_FULL
    print_info "Full mode: Using $EPOCHS epochs (production)"
fi

# Display configuration
echo ""
echo "=================================="
echo "  TGVAnn Experiment Configuration"
echo "=================================="
echo "Dataset:       $DATASET"
echo "Classes:       $NUM_CLASSES"
echo "RGB Dir:       $RGB_DIR"
echo "Texture Dir:   $TEXTURE_DIR"
echo "Output Dir:    $OUTPUT_DIR"
echo "Batch Size:    $BATCH_SIZE"
echo "Epochs:        $EPOCHS"
echo "Learning Rate: $LR"
echo "Mode:          $MODE"
echo "=================================="
echo ""

# Function to generate textures
generate_textures() {
    print_info "Generating LDP texture features for $DATASET..."
    
    if [ "$DATASET" == "maize" ]; then
        python $TEXTURE_SCRIPT
    else
        python $TEXTURE_SCRIPT
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Texture generation complete!"
    else
        print_error "Texture generation failed!"
        exit 1
    fi
}

# Function to train model
train_model() {
    print_info "Training TGVAnn on $DATASET dataset..."
    
    python $TRAIN_SCRIPT \
        --rgb_dir $RGB_DIR \
        --texture_dir $TEXTURE_DIR \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --scheduler cosine \
        --augment \
        --save_freq 10
    
    if [ $? -eq 0 ]; then
        print_success "Training complete!"
        print_info "Best model saved to: $OUTPUT_DIR/best_model*.pth"
    else
        print_error "Training failed!"
        exit 1
    fi
}

# Function to evaluate model
evaluate_model() {
    print_info "Evaluating model on $DATASET dataset..."
    
    # Find best model checkpoint
    if [ "$DATASET" == "maize" ]; then
        CHECKPOINT="$OUTPUT_DIR/best_model.pth"
    else
        CHECKPOINT="$OUTPUT_DIR/best_model_sugarcane.pth"
    fi
    
    if [ ! -f "$CHECKPOINT" ]; then
        print_error "Checkpoint not found: $CHECKPOINT"
        print_warning "Please train the model first!"
        exit 1
    fi
    
    python $EVAL_SCRIPT \
        --rgb_dir $RGB_DIR \
        --texture_dir $TEXTURE_DIR \
        --checkpoint $CHECKPOINT \
        --output_dir ./eval_results/$DATASET \
        --batch_size $BATCH_SIZE
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation complete!"
        print_info "Results saved to: ./eval_results/$DATASET/"
    else
        print_error "Evaluation failed!"
        exit 1
    fi
}

# Function to generate Grad-CAM
generate_gradcam() {
    print_info "Generating Grad-CAM visualizations for $DATASET..."
    
    # Find best model checkpoint
    if [ "$DATASET" == "maize" ]; then
        CHECKPOINT="$OUTPUT_DIR/best_model.pth"
    else
        CHECKPOINT="$OUTPUT_DIR/best_model_sugarcane.pth"
    fi
    
    if [ ! -f "$CHECKPOINT" ]; then
        print_error "Checkpoint not found: $CHECKPOINT"
        print_warning "Please train the model first!"
        exit 1
    fi
    
    python $GRADCAM_SCRIPT \
        --rgb_dir $RGB_DIR \
        --texture_dir $TEXTURE_DIR \
        --checkpoint $CHECKPOINT \
        --output_dir ./gradcam_results/$DATASET \
        --mode all \
        --num_samples 15
    
    if [ $? -eq 0 ]; then
        print_success "Grad-CAM generation complete!"
        print_info "Visualizations saved to: ./gradcam_results/$DATASET/"
    else
        print_error "Grad-CAM generation failed!"
        exit 1
    fi
}

# Main execution
echo ""
print_info "Starting TGVAnn experiment: $DATASET - $ACTION"
echo ""

case $ACTION in
    texture)
        generate_textures
        ;;
    train)
        train_model
        ;;
    eval)
        evaluate_model
        ;;
    gradcam)
        generate_gradcam
        ;;
    all)
        print_info "Running complete pipeline..."
        echo ""
        
        # Step 1: Generate textures
        print_info "Step 1/4: Texture Generation"
        generate_textures
        echo ""
        
        # Step 2: Train model
        print_info "Step 2/4: Model Training"
        train_model
        echo ""
        
        # Step 3: Evaluate model
        print_info "Step 3/4: Model Evaluation"
        evaluate_model
        echo ""
        
        # Step 4: Generate Grad-CAM
        print_info "Step 4/4: Grad-CAM Visualization"
        generate_gradcam
        echo ""
        
        print_success "Complete pipeline finished successfully!"
        ;;
    *)
        print_error "Unknown action: $ACTION"
        usage
        ;;
esac

echo ""
print_success "Experiment complete!"
echo ""
echo "Summary:"
echo "--------"
echo "Dataset:    $DATASET ($NUM_CLASSES classes)"
echo "Action:     $ACTION"
echo "Output:     $OUTPUT_DIR"
echo ""

if [ "$ACTION" == "all" ] || [ "$ACTION" == "train" ]; then
    echo "Next steps:"
    echo "  1. Check training curves: $OUTPUT_DIR/training_history*.png"
    echo "  2. View evaluation results: ./eval_results/$DATASET/"
    echo "  3. Explore Grad-CAM: ./gradcam_results/$DATASET/"
    echo ""
fi

print_info "Done! 🎉"