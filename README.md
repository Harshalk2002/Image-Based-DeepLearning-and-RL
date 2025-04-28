# Image-Based Deep Learning & Reinforcement Learning Strategy Design

## Team 6 Members
- Pamela Alvarado-Zarate
- Mark Hite
- Harshal Kamble
- Vishnu Sankhyan
- Venkatesh Subramony

## Project Overview
This project consists of two major parts:

- **Part A: Image-Based Deep Learning**
  - Image Generation using GANs
  - Object Detection using YOLOv5
  - Image Captioning using a CNN-RNN Encoder-Decoder Architecture

- **Part B: Reinforcement Learning Strategy Design**
  - Design and development of a Deep Q-Learning strategy for a 2D Street Racer game.

## Part A: Image-Based Deep Learning

### 1. Image Generation (GAN)
- **Dataset:** COCO 2017 (Common Objects in Context), subset of 100,000 images.
- **GAN Architecture:**
  - **Generator:** 5 convolutional layers, ReLU activations, Tanh output.
  - **Discriminator:** 5 convolutional layers, Leaky ReLU activations, Sigmoid output.
- **Training:**
  - 100,000 images resized to 64x64.
  - 20 epochs.
  - Adam optimizer with learning rate 0.0002.

### 2. Object Detection (YOLOv5)
- **Model:** Pre-trained YOLOv5s model.
- **Inference:** 5 custom images (Dog, Bird, Car, Street, Lemur).
- **Threshold:** 0.25 confidence threshold.
- **Post-processing:** Non-Maximum Suppression (NMS) applied.

### 3. Image Captioning
- **Encoder:** ResNet-50 pretrained on ImageNet.
- **Decoder:** RNN with LSTM (512 hidden units).
- **Hyperparameters:**
  - Max caption length: 34 tokens.
  - Vocabulary size: 5,000 words.
  - Special tokens: `<start>`, `<end>`, `<unk>`.
- **Training:**
  - Teacher forcing.
  - Adam optimizer.
  - 64 mini-batches over 30 epochs.

## Part B: Reinforcement Learning Strategy Design

- **Game:** 2D Street Racer (Top-down endless racing game).
- **States:** Lane position, obstacle positions, speed, distance traveled, coin locations.
- **Actions:** Move Left, Move Right, Accelerate, Decelerate, Do Nothing.
- **Rewards:**
  - +1 for collecting coins.
  - +5 for distance milestones.
  - -10 for collisions.
  - -5 for moving off-road.
- **Model:**
  - Deep Q-Network (DQN).
  - Fully connected neural network (2 hidden layers of 64 neurons each).
- **Training Approach:**
  - Epsilon-greedy exploration.
  - Experience replay.
  - Target network stabilization.

## Files
- `GAN_final.ipynb` – Full Jupyter Notebook for Part A.
- `imaging.py` – Core code for Object Detection and Image Captioning.
- `Report.docx` – Full final project report.
- `Team 6 - Final Project.pptx` – Presentation slides.

## Instructions

1. **Image Generation (GAN):**
   - Run `GAN_final.ipynb`.
   - Train the GAN on COCO dataset subset.

2. **Object Detection:**
   - Upload 5 custom images.
   - Run YOLOv5s model inference via `imaging.py` or `GAN_final.ipynb`.

3. **Image Captioning:**
   - Use encoder-decoder model to generate captions for uploaded images.

4. **Reinforcement Learning:**
   - Follow strategy outlined in `Report.docx` for DQN agent design.

## Results Summary

- GAN successfully generated realistic 64x64 synthetic images after 20 epochs.
- YOLOv5 accurately detected multiple objects including dogs, birds, cars, and pedestrians.
- Captioning model generated repetitive but partially contextually accurate captions; future improvements recommended.
- RL Strategy was outlined with clear modeling of states, actions, rewards, and learning flow.

---

> **Disclaimer:** The analysis and results are based on synthetically generated data. This is NOT a reflection of Truist's data.

---

## Acknowledgments
Special thanks to the faculty and course instructors for guidance throughout this project.

## Contact
For any inquiries or collaboration, please contact any of the team members listed above.

---

# Thank you!
