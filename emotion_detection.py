from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision import transforms
from emotionRecog import EmotionNet
from emotionRecog import resNet
import torch.nn.functional as nnf
from emotionRecog import utils
import numpy as np
import argparse
import torch
import cv2
 
# the argument parser and the arguments required
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True,
                    help="path to the trained model")
parser.add_argument("-n", "--network", type=str, required=True,
                    help="which network to use: VGG11/VGG13/VGG16/VGG19/resnet")
parser.add_argument('-p', '--prototxt', type=str, required=True,
                    help='Path to deployed prototxt.txt model architecture file')
parser.add_argument('-c', '--caffemodel', type=str, required=True,
                    help='Path to Caffe model containing the weights')
parser.add_argument("-conf", "--confidence", type=int, default=0.5,
                    help="the minimum probability to filter out weak detection")
args = vars(parser.parse_args())

# load our serialized model from disk for face detection
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['caffemodel'])
 

device = "cuda" if torch.cuda.is_available() else "cpu"
 
# dictionary mapping for different outputs
emotion_dict = {0: "Angry", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral",
                6: "Sad", 7: "Surprised"}


network_type = args['network']
if(network_type == "resnet"):
    model = resNet.ResNet18()
else:
    model = EmotionNet(num_of_channels=1, num_of_classes=8,net=network_type)
    
model_weights = torch.load(args["model"])
model.load_state_dict(model_weights)
model.to(device)
model.eval()
 
# a list of preprocessing steps to apply on each image during runtime
data_transform = transforms.Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),
    Resize((48, 48)),
    ToTensor()
])


# capture the camera
vs = cv2.VideoCapture(0)
 

while True:
 
    (grabbed, frame) = vs.read()
 
    if not grabbed:
        break
 
    frame = utils.resize_image(frame, width=800, height=800)
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # initialize an empty canvas to output the probability distributions
    canvas = np.zeros((450, 350, 3), dtype="uint8")
 
    # get the frame dimension, resize it and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))
 
    # infer the blog through the network to get the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
 
        # grab the confidence associated with the model's prediction
        confidence = detections[0, 0, i, 2]
 
        # eliminate weak detections, ensuring the confidence is greater
        # than the minimum confidence pre-defined
        if confidence > args['confidence']:
 
            # compute the (x,y) coordinates (int) of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            # grab the region of interest within the image (the face)
            face = frame[start_y:end_y, start_x:end_x]
            face = data_transform(face)
            face = face.unsqueeze(0)
            face = face.to(device)
 
            # compute the probability score and class for each face and grab the readable emotions
            predictions = model(face)
            prob = nnf.softmax(predictions, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            top_p, top_class = top_p.item(), top_class.item()
 
            # grab the list of predictions along with their associated labels
            emotion_prob = [p.item() for p in prob[0]]
            emotion_value = emotion_dict.values()
            # draw the probability distribution on an empty canvas initialized
            for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
                prob_text = f"{emotion}: {prob * 100:.2f}%"
                width = int(prob * 350)
                cv2.rectangle(canvas, (5, (i * 50) + 5), (width, (i * 50) + 50),
                              (255, 0, 0), -1)
                cv2.putText(canvas, prob_text, (5, (i * 50) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
            # draw the bounding box of the face along with the associated emotion
            # and probability
            face_emotion = emotion_dict[top_class]
            face_text = f"{face_emotion}: {top_p * 100:.2f}%"
            cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(output, face_text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.05, (0, 255, 0), 2)
 
    # display the output to our screen
    cv2.imshow("Face", output)
    cv2.imshow("Emotion probability distribution", canvas)
 
    # break the loop if the `q` key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
cv2.destroyAllWindows()
vs.release()