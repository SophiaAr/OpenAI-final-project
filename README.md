# OpenAI-final-project

Introducing the GoalGrid Environment project that was inspired by the grounded language approach (here is the reference paper https://arxiv.org/abs/1706.06551). This project was executed within the OpenAI scholarship program. The GoalGridWorld environment is a matrix where an agent navigates followed by commands  [go, avoid] to hit target cells that are represented as randomly displaced pairs of color objects [triangle, square, circle/red, green, blue]. Basically, there are 18 possible commands but only 8 actual "words". Thanks to an NLP-like approach in this case, an agent can handle more complex/combinatorial commands. We removed the [blue, circle] part from the training process to see if during the inference an agent can generalize and ‘recognize’ the target cell presented by [blue, circle] pair even if it never saw it during the training process. 

Feel free to use the code to run your experiments. 

I want to thank Alec Radford for his mentorship and introducing me to NLP and Joshua Achiam for his valuable advice on reinforcement learning and creating the Spinning Up (https://spinningup.openai.com/en/latest/index.html) that was extremely helpful for my project.

