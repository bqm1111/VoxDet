import torch
import open_clip


def extract_clip_features(labels: list, model_name="ViT-B-32", pretrained="openai"):
    print("Loading CLIP {} model".format(model_name))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    print("Finish loading")
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.cuda()  # remove if CPU-only

    text_tokens = tokenizer(labels)  # shape: [N, context_length]
    text_tokens = text_tokens.cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


if __name__ == "__main__":
    from PIL import Image

    image = Image.open("/home/minh/Pictures/images.jpeg").convert("RGB")

    labels = ["car", "truck", "road", "bus", "face", "dog"]
    model_name = "ViT-B-32"
    pretrained="openai"
    print("Loading CLIP {} model".format(model_name))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    print("Finish loading")
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.cuda()  # remove if CPU-only
    image_tensor = preprocess(image).unsqueeze(0).to("cuda")

    text_tokens = tokenizer(labels)  # shape: [N, context_length]
    text_tokens = text_tokens.cuda()

    import time
    start = time.time()

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    print("time elapsed = ", time.time() - start)
    
    # txt_features = extract_clip_features(labels)
    print(text_features)
    print(text_features.shape)
    print("image feature shape = ", image_features.shape)
    cos_sim = text_features @ image_features.T
    cos_sim = cos_sim.squeeze(1)
    print(cos_sim)
    
    