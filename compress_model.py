from keras.models import load_model

model=load_model("drowsiness.h5")

model.save("drowsiness.keras")

print("New model saved")
