import matplotlib.pyplot as plt
from pathlib import Path
import csv
import json



epoch = 1
epochs = []
r1 = []
r2 = []
rl = []
for line in Path("./log/result.json").read_text().split("\n")[:-1]:
    data = json.loads(line)
    if "eval_loss" in data.keys():
        epochs.append(epoch)
        epoch += 1
        r1.append(data["eval_rouge-1_f"])
        r2.append(data["eval_rouge-2_f"])        
        rl.append(data["eval_rouge-l_f"])

plt.plot(
    epochs,
    r1,
    color="orange",
    label="rouge-1",
)
plt.plot(
    epochs,
    r2,
    color="green",
    label="rouge-2",
)
plt.plot(
    epochs,
    rl,
    color="red",
    label="rouge-l",
)
plt.title("rouge score")
plt.xlabel("epoch")
plt.legend()
plt.savefig("rouge.jpg")
plt.clf()