python3 export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ssd_mobilenet_v1_pets.config \
        --trained_checkpoint_prefix training/model.ckpt-2588 \
        --output_directory trained_model
