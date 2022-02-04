import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Button, Label, Entry
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pyttsx3
from sklearn.metrics import plot_confusion_matrix

# open a file browser window to select data file.


def file_select():
    file_path = filedialog.askopenfilename()
    return file_path

# set path of training x values


def set_trx_path():
    global trxPath, trxlabel
    trxPath = file_select()
    trxlabel.config(text=trxPath)

# set path of training y values


def set_try_path():
    global tryPath, trylabel
    tryPath = file_select()
    trylabel.config(text=tryPath)

# set path of testing x values


def set_tex_path():
    global texPath, texlabel
    texPath = file_select()
    texlabel.config(text=texPath)

# set path of testing y values


def set_tey_path():
    global teyPath, teylabel
    teyPath = file_select()
    teylabel.config(text=teyPath)

# set parameters of the neural network


def set_variable():
    global neuron_label, neuron_num, LR_label, LR, main
    LR = LR_label.get()
    neuron_num = neuron_label.get()
    main.destroy()


# main window to select files and set NN parameters.
global trxPath, tryPath, texPath, teyPath, trxlabel, trylabel, texlabel, teylabel, LR, neuron_num, LR_label, neuron_label, main
main = tk.Tk()
main.geometry("600x600")
main.title('PA2')

Label(text="Training data:").grid(row=0, column=0, pady=20)
trxlabel = Label(bg='white', width=50)
trxlabel.grid(row=0, column=1, padx=5)
trxButton = Button(main, text='Browse', command=set_trx_path)
trxButton.grid(row=0, column=2)

Label(text="Training target:").grid(row=1, column=0, pady=20)
trylabel = Label(bg='white', width=50)
trylabel.grid(row=1, column=1, padx=5)
tryButton = Button(main, text='Browse', command=set_try_path)
tryButton.grid(row=1, column=2)

Label(text="Testing data:").grid(row=2, column=0, pady=20)
texlabel = Label(bg='white', width=50)
texlabel.grid(row=2, column=1, padx=5)
texButton = Button(main, text='Browse', command=set_tex_path)
texButton.grid(row=2, column=2)

Label(text="Testing target:").grid(row=3, column=0, pady=20)
teylabel = Label(bg='white', width=50)
teylabel.grid(row=3, column=1, padx=5)
teyButton = Button(main, text='Browse', command=set_tey_path)
teyButton.grid(row=3, column=2)

Label(text="Number of hidden neurons:").grid(row=4, column=0, pady=20)
neuron_label = Entry(main, width=50)
neuron_label.insert(-1, '50')
neuron_label.grid(row=4, column=1)
neuron_num = neuron_label.get()

Label(text="Learning rate:").grid(row=5, column=0, pady=20)
LR_label = Entry(main, width=50)
LR_label.insert(-1, '0.1')
LR_label.grid(row=5, column=1)
LR = LR_label.get()

doneButton = Button(main, text='Train!',
                    command=set_variable, width=12, height=3)
doneButton.grid(row=6, column=1)
main.mainloop()

# function to read x values for both training and testing


def get_data(path):
    file = open(path)
    mylist = []
    while 1:
        char = file.read(1)
        if not char:
            break
        if char.isdigit():
            mylist.append(int(char))
    file.close()
    return mylist

# function to read y values for both training and testing


def get_targets(path):
    my_list = []
    with open(path) as f:
        for line in f:
            if line.strip().isdigit():
                my_list.append(int(line.strip()))
    return my_list


# reshaping data in a NumPy Array to allow model creation
tr_x = get_data(trxPath)
tr_x = np.array(tr_x).reshape(6000, 256)

tr_y = get_targets(tryPath)
tr_y = np.array(tr_y).reshape(6000, 1)

te_x = get_data(texPath)
te_x = np.array(te_x).reshape(5000, 256)

te_y = get_targets(teyPath)
te_y = np.array(te_y).reshape(5000, 1)

# initializing neural network with entered parameters
clf = MLPClassifier(
    hidden_layer_sizes=(int(neuron_num),),
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=float(LR),
)

root = tk.Tk()
root.geometry("1500x1000")
root.title('PA2 Graphs')
root.config(bg='white')


# training model and calculating training time.
start_time = float(time.time())
clf.fit(tr_x, tr_y)
end_time = float(time.time())
train_time = 'Training time = %.4f' % (end_time - start_time)

# plotting error curve
fig1 = plt.figure(dpi=100)
ax = fig1.add_subplot(111)
line1 = FigureCanvasTkAgg(fig1, root)
line1.get_tk_widget().grid(row=0, column=0)
ax.set(title='Error Curve', ylabel='Error', xlabel='Epochs')
plt.plot(clf.loss_curve_)

# testing the model with testing data and plotting confusion matrix.
fig2 = plt.figure(figsize=(9, 9))
ax2 = fig2.add_subplot(111)
line2 = FigureCanvasTkAgg(fig2, root)
line2.get_tk_widget().grid(row=0, column=1)
plot_confusion_matrix(clf, te_x, te_y, ax=ax2)

acc = "Overall Accuracy: %f" % clf.score(te_x, te_y)
print()
Label(root, text=acc).grid(row=1, column=1)
Label(root, text=train_time).grid(row=1, column=0)
drawButton = Button(root, text='Draw!',
                    command=root.destroy, width=12, height=3)
drawButton.grid(row=2, column=0)
root.mainloop()


# drwaing segment BONUS
ink = []


class BT():
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.btn = Button(window, bg='white', width=4,
                          height=2, command=self.wall_btn)
        self.btn.grid(row=i, column=j)

    def wall_btn(self):
        if self.btn['bg'] == 'white':
            self.btn.configure(bg="Black")
            ink.append((self.j, self.i))
            temp = self.j
            for x in range(self.i):
                temp += 16
            code[temp] = 1
        else:
            self.btn.configure(bg="white")


window = tk.Tk()
window.title("PA1")
Width = 500/16

code = []


def render_grid():
    global Width, player, end
    for i in range(16):
        for j in range(16):
            BT(i, j)
            code.append(0)


render_grid()


def start_game():
    window.mainloop()


start_game()
code = np.array(code).reshape(1, 256)
prediction = "the number is " + np.array2string(clf.predict(code))
print(prediction)
engine = pyttsx3.init()
engine.say(prediction)
engine.runAndWait()
