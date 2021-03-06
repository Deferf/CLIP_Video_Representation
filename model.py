import numpy as np
import cv2
from PIL import Image
import torch

def encode_video(video_file, preprocess, model, resolution, image_mean, image_std):
  cap = cv2.VideoCapture(video_file)
  frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  images = []
  fc = 0
  ret = True

  while (fc < frameCount  and ret):
      ret, frame = cap.read()
      if not ret: # if file is empty break loop
        break
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
      fc += 1
  
  image_input = torch.tensor(np.stack(images)).cuda()
  image_input -= image_mean[:, None, None]
  image_input /= image_std[:, None, None]

  with torch.no_grad():
    image_features = model.encode_image(image_input).float()

  image_features /= image_features.norm(dim=-1, keepdim=True)
  cap.release()

  return image_features


def video_to_tensor(video_file, preprocess, each = 0, use_fps = False):
  # Samples a frame each X frames.
  torch.device("cuda:0")
  cap = cv2.VideoCapture(video_file)
  frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  images = []
  fc = 0
  ret = True
  #print(fps, "fps")

  if use_fps:
    each = fps

  skip_counter = each

  included = []

  while (fc < frameCount  and ret):
    if skip_counter >= each:
      ret, frame = cap.read()
      skip_counter = 0
      if not ret: # if file is empty break loop
          break
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
      included.append(fc)

    skip_counter += 1

    fc += 1

  cap.release()
  #print(included)
  return torch.tensor(np.stack(images)).cuda()

def gen_video_encoder(preprocess, model, resolution, image_mean, image_std):

  def encode_video(video_file, each = 0):
    image_input = video_to_tensor(video_file, preprocess, each)
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    with torch.no_grad():
      image_features = model.encode_image(image_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features

  return encode_video

def encode_text(tokenizer, texts):
  text_tokens = [tokenizer.encode(desc) for desc in texts]
  text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']

  for i, tokens in enumerate(text_tokens):
    tokens = [sot_token] + tokens + [eot_token]
    text_input[i, :len(tokens)] = torch.tensor(tokens)

  text_input = text_input.cuda()
  with torch.no_grad():
      text_features = model.encode_text(text_input).float()
  text_features /= text_features.norm(dim=-1, keepdim=True)

  return text_features


def gen_text_encoder(tokenizer, model, context_len):
  def custom_encode_text(texts):
    clip = context_len - 2 # generally context_len is 77, we substract 2 because we need 2 start and end tokens
    text_tokens = [tokenizer.encode(desc)[:clip] for desc in texts] # we will clip the sentences 
    text_input = torch.zeros(len(text_tokens), context_len, dtype=torch.long)
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    for i, tokens in enumerate(text_tokens):
      tokens = [sot_token] + tokens + [eot_token]
      token_tensor = torch.tensor(tokens)
      text_input[i, :len(tokens)] = token_tensor

    text_input = text_input.cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features
  return custom_encode_text
