# OpenAI-final-project

Introducing the GoalGridWorld Environment project that was inspired by the grounded language learning (here is the reference paper https://arxiv.org/abs/1706.06551). This project was executed within the OpenAI scholarship program. The GoalGridWorld environment is a matrix where an agent navigates according to commands [go, avoid] to hit or avoid target cells that are represented as randomly displaced pairs of colored objects [triangle, square, circle/red, green, blue]. Basically, there are 18 possible commands but only 8 actual "words". Thanks to an NLP-like approach in this case, an agent can handle more complex/combinatorial commands. 

Feel free to use the code to run your experiments. 

I want to thank Alec Radford for his mentorship and introducing me to NLP and Joshua Achiam for his valuable advice on reinforcement learning and creating the Spinning Up (https://spinningup.openai.com/en/latest/index.html) that was extremely helpful for my project.


# Setup

* conda install --yes --file requirements.txt
* Source activate spinningup
* Python3 train.py 


