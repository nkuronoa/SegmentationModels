import segmentation_models_pytorch as smp


def load_segment_model(modeltype: str, encoder: str, encoder_weights: str, n_classes: int, activation: str):
    if(modeltype == "UNet++"):
        model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes, activation=activation)
    elif(modeltype == "DeepLabV3+"):
        model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes, activation=activation)
    else:
        model = smp.DeepLabV3(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes, activation=activation)

    return model
