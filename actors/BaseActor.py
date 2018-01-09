

'''
Some actors may need a history, some may need an episode bank.

Either way, after the actor will do what he needs when 'act' is called, he will return a state to continue upon.
This could be the next state continuing the same game, or a replayed experience from his memory.

We check after every action if the player has won. We collect all finished games and apply
when hitting the minibatch size.

'''

class BaseActor(object):
    def __init__(self):
        pass

    ' Must some state to continue with (eg. next state, random episode from some history...) '
    def act(self, gstate, model):
        pass

    def maybe_print(self):
        raise Exception('implement')
