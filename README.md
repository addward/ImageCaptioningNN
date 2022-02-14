# ImageCaptioningNN
This repo is created as final project in [Deep Learning School](https://en.dlschool.org/).<br>
It consists of two models, [baseline](#architecture-of-the-base-model) and [advanced model](#architecture-of-the-advanced-model) with additive attention. <br>
All the training and inference are presented in *.ipynb files. One for base mode and another for advanced.
## Architecture of the base model

<img src="https://live.staticflickr.com/65535/51876877412_9d994fd916_k.jpg" alt="drawing" width="800"/>

### Results
#### Tony Hawk
<img src="https://github.com/addward/ImageCaptioningNN/blob/main/img_examples/tony_hawk.jpg" alt="drawing" width="300"/>
Greedy sentence (choose the most probable word on each generation step):

* a skateboarder is doing a trick on a ramp .

Sampling (Choose words using the probablity distribution from the NN):

* a skateboarder is doing a trick on a ramp .
* a skateboarder is doing a trick on a ramp .
* a man riding a skateboard on top of a cement ramp .
* a man on a skateboard doing a trick on a ramp .
* a skateboarder is doing a trick on a ramp .

Example of sampling:

<img src="https://live.staticflickr.com/65535/51879915030_efa55d79c2_o.png" alt="drawing" width="1500"/>

### Soccer

<img src="https://github.com/addward/ImageCaptioningNN/blob/main/img_examples/soccer.jpg" alt="drawing" width="300"/>
Greedy sentence:

* a group of people playing a game of soccer .

Sampling :

* a group of people playing a game of soccer .
* a group of people playing a game of tennis .
* a group of people standing on a tennis court .
* a group of people playing tennis on a court
* a group of people standing on a tennis court .

## Architecture of the advanced model

<img src="https://live.staticflickr.com/65535/51878568010_343866e0e5_o.jpg" alt="drawing" width="800"/>

### Results
#### Cyclist 
<img src="https://github.com/addward/ImageCaptioningNN/blob/main/img_examples/cycler.jpg" alt="drawing" width="300"/>
Greedy sentence :

* a man riding a bike down a street .<br>
attention weights:
<img src="https://github.com/addward/ImageCaptioningNN/blob/main/img_examples/att_cyclist.png" alt="drawing" width="300"/>

Sampling :

* a man riding a bike down a street .
* a man riding a bike down a street .
* a woman riding a bike down a street .
* a man riding a bike down a street .
* a person riding a bike on a street .

#### A Cat with a dog 
<img src="https://github.com/addward/ImageCaptioningNN/blob/main/img_examples/cat_dog.jpg" alt="drawing" width="300"/>
Greedy sentence :

* a cat is sitting on a couch .<br>
attention weights:
<img src="https://github.com/addward/ImageCaptioningNN/blob/main/img_examples/att_cat_dog.png" alt="drawing" width="300"/>

Sampling :

* a black and white cat sitting on a table
* a black and white cat is next to a white cat .
* a cat that is sitting on a couch
* a black and white cat is sitting in a box
* a black and white cat is standing in a room .

# How to run telegram bot
1. Recieve token from [Bot Father](https://t.me/botfather)
2. Type the recieved token into start.py file
3. Start start.py script (sudo python3 start.py)

# References
1. https://habr.com/ru/post/316666/
2. https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
3. https://www.tensorflow.org/tutorials/text/image_captioning
4. https://medium.com/towards-data-science/image-captioning-in-deep-learning-9cd23fb4d8d2
5. https://en.dlschool.org/
6. https://docs.h5py.org/en/stable/
