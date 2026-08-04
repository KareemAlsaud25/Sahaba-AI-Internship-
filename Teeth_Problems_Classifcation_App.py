import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load("best_resnet50.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Teeth Disease Classifier")
st.write("Upload a dental image to classify it into one of 7 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    st.subheader(f"Prediction: {class_names[predicted_idx]}")
    st.write(f"Confidence: {probabilities[predicted_idx].item()*100:.2f}%")

    st.bar_chart({class_names[i]: probabilities[i].item() for i in range(len(class_names))})