To run the code, please install requirements in requirements.txt and run the code 

``
python deepQ.py
``

The code is set up to run multiple times to generate mean and std learning curves.

An example learning curve showing score vs training episode is shown in

``
example_learning_curve.png
``

And is the cumilation of 7 runs of the model, showing mean and standard deviation of score vs training episode (which admittedly isn't a good measure, but shows how variable the results can be).

I run the model up until 400 episodes at which point it normally learns to play the game well. More episodes would improve this further but takes a long time for me to run.