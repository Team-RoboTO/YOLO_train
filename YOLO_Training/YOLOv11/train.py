from ultralytics import YOLO, checks, hub
checks()

hub.login('b4931f21c62cfc7ead7513cd48bf10552c1efa489e')

model = YOLO('https://hub.ultralytics.com/models/79cacTfQJRM0SSVrd7Fh')
results = model.train(name='yolov11_0.001lr_SGD_coslr')