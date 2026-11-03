import os

train_count=5000
test_count=1000

classes=["Open-Eyes","Close-Eyes"]

for c in classes:
    path="train/"+c
    files=os.listdir(path)

    for f in files[train_count:]:
        os.remove(path+"/"+f)

    path="test/"+c
    files=os.listdir(path)

    for f in files[test_count:]:
        os.remove(path+"/"+f)

print("Extra images removed")
