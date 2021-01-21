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