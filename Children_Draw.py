
import os
import gdown
import streamlit as st

import torch

import torch.nn as nn

from PIL import Image

from torchvision import models

from torchvision.models import ResNet18_Weights


# ====================================================
# 1. MODEL PATH
# ====================================================

#MODEL_PATH = r"C:\Users\Engy\Downloads\best_model.pth"
MODEL_PATH = "best_model.pth"

GOOGLE_DRIVE_FILE_ID = "13RFARSgdKXkXwggmbpLxR-7gZl5aSDmq"

if not os.path.exists(MODEL_PATH):

    gdown.download(
        f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )




st.markdown(
    """
    <style>
    .stApp {
        background-color: #EAF6FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ====================================================
# 2. CLASS NAMES
# ====================================================

classes = [
    "Angry",
    "Fear",
    "Happy",
    "Sad"
]


# ====================================================
# 3. CREATE RESNET18 MODEL
# ====================================================

model = models.resnet18(
    weights=None
)


# Trial 5 dropout
DROPOUT = 0.45185613635781285


# Change the final layer
# from ImageNet classes to our 4 emotions

model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(
        model.fc.in_features,
        4
    )
)


# ====================================================
# 4. CHECK MODEL FILE
# ====================================================

if not os.path.exists(MODEL_PATH):

    st.error("Model file was not found.")

    st.stop()


# ====================================================
# 5. LOAD TRAINED RESNET18 MODEL
# ====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ====================================================
# 6. EVALUATION MODE
# ====================================================

model.eval()


# ====================================================
# 7. IMAGE TRANSFORMATION
# ====================================================

weights = ResNet18_Weights.DEFAULT

transform = weights.transforms()


# ====================================================
# 8. STREAMLIT TITLE
# ====================================================

st.title("Child Drawing Analyzer")

st.write(
    "Upload a child's drawing to analyze its "
    "predicted emotional expression."
)


# ====================================================
# 9. UPLOAD IMAGE
# ====================================================

uploaded_file = st.file_uploader(
    "Choose a drawing",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ====================================================
# 10. IF IMAGE WAS UPLOADED
# ====================================================

if uploaded_file is not None:

    # Open image

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image

    st.image(
        image,
        caption="Uploaded Drawing",
        use_container_width=True
    )


    # ====================================================
    # 11. ANALYZE BUTTON
    # ====================================================

    if st.button("Analyze Drawing"):


        # ====================================================
        # 12. CONVERT IMAGE TO TENSOR
        # ====================================================

        input_image = transform(image)


        # ====================================================
        # 13. ADD BATCH DIMENSION
        # ====================================================

        input_image = input_image.unsqueeze(0)


        # ====================================================
        # 14. MODEL PREDICTION
        # ====================================================

        with torch.no_grad():

            outputs = model(
                input_image
            )


        # ====================================================
        # 15. CONVERT OUTPUTS TO PROBABILITIES
        # ====================================================

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # ====================================================
        # 16. FIND HIGHEST PROBABILITY
        # ====================================================

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()


        # Get emotion name

        predicted_emotion = classes[
            predicted_index
        ]


        # ====================================================
        # 17. GET CONFIDENCE
        # ====================================================

        confidence = probabilities[
            0,
            predicted_index
        ].item()


        confidence_percentage = (
            confidence * 100
        )


        # ====================================================
        # 18. DISPLAY AI RESULT
        # ====================================================

        st.header("AI Prediction")

        st.success(
            f"Predicted Emotion: {predicted_emotion}"
        )

        st.write(
            f"Confidence: "
            f"{confidence_percentage:.2f}%"
        )


        # ====================================================
        # 19. PSYCHOLOGY-ORIENTED REPORT
        # ====================================================

        st.header(
            "Psychology-Oriented Observation"
        )


        reports = {

            "Happy": {

                "observation": (
                    "The drawing shows an emotional "
                    "expression associated with happiness "
                    "or positive emotion."
                ),

                "possible_meaning": (
                    "This may reflect feelings such as "
                    "happiness, enjoyment, comfort, "
                    "excitement, or a positive experience."
                ),

                "support": (
                    "You can encourage the child to talk "
                    "about the drawing and ask what they "
                    "enjoyed or what made them feel happy."
                )

            },


            "Sad": {

                "observation": (
                    "The drawing shows an emotional "
                    "expression associated with sadness."
                ),

                "possible_meaning": (
                    "This may reflect feelings of sadness, "
                    "disappointment, loneliness, missing "
                    "someone, or an unpleasant experience."
                ),

                "support": (
                    "Consider gently asking the child "
                    "about the drawing and giving them "
                    "an opportunity to express how they "
                    "feel without judging or pressuring them."
                )

            },


            "Fear": {

                "observation": (
                    "The drawing shows an emotional "
                    "expression associated with fear or worry."
                ),

                "possible_meaning": (
                    "This may reflect feelings such as "
                    "worry, nervousness, uncertainty, "
                    "or something the child may perceive "
                    "as frightening."
                ),

                "support": (
                    "A parent or caregiver can calmly ask "
                    "the child about the drawing and what "
                    "part of it they find important or frightening."
                )

            },


            "Angry": {

                "observation": (
                    "The drawing shows an emotional "
                    "expression associated with anger "
                    "or frustration."
                ),

                "possible_meaning": (
                    "This may reflect feelings of frustration, "
                    "disagreement, irritation, or an experience "
                    "that made the child upset."
                ),

                "support": (
                    "It may help to give the child a safe "
                    "opportunity to describe what happened "
                    "and express their feelings."
                )

            }

        }


        # Get the report for the predicted emotion

        report = reports[
            predicted_emotion
        ]


        # ====================================================
        # 20. OBSERVATION
        # ====================================================

        st.subheader(
            "Observation"
        )

        st.write(
            report["observation"]
        )


        # ====================================================
        # 21. POSSIBLE EMOTIONAL MEANING
        # ====================================================

        st.subheader(
            "Possible Emotional Meaning"
        )

        st.write(
            report["possible_meaning"]
        )


        # ====================================================
        # 22. SUGGESTED SUPPORT
        # ====================================================

        st.subheader(
            "Suggested Support"
        )

        st.write(
            report["support"]
        )


        # ====================================================
        # 23. DISCLAIMER
        # ====================================================

        st.info(
            "Important: This AI result describes the "
            "emotional expression classified from the drawing. "
            "It is not a psychological diagnosis. Children's "
            "drawings can have many meanings and should be "
            "considered together with the child's behavior, "
            "communication, and personal context."
        )

