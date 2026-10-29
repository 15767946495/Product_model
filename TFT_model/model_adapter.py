"""TFT 模型的轻量构建边界。"""

from .models import TFTEncoderForYieldPrediction


def build_model(model_config: dict) -> TFTEncoderForYieldPrediction:
    return TFTEncoderForYieldPrediction(
        soil_dim=model_config.get("soil_dim", 7),
        dynamic_feature_names=model_config["dynamic_feature_names"],
        hidden_size=model_config["hidden_size"],
        num_lstm_layers=model_config.get("num_lstm_layers", 1),
        dropout=model_config.get("dropout", 0.3),
        output_size=model_config.get("output_size", 1),
        num_heads=model_config.get("num_heads", 3),
        vit_freeze_backbone=model_config.get("vit_freeze_backbone", False),
        vit_pretrained=model_config.get("vit_pretrained", False),
        pvt_pretrained_path=model_config.get("pvt_pretrained_path"),
        remote_constant=model_config.get("remote_constant", False),
        rs_as_variable=model_config.get("rs_as_variable", False),
        rs_aux=model_config.get("rs_aux", False),
        rs_feature_dim=model_config.get("rs_feature_dim", 0),
    )
