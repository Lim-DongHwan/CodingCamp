# commands/base_command.py
import os
from typing import List

"""
TODO 3-1: The BaseCommand class has a show_usage method implemented, but the execute method is not 
implemented and is passed on to the child class. Think about why this difference is made.

Answer (You may write your answer in either Korean or English):
command의 usage에 대한 설명은 하위의 child class 전부에게 필요할 수 있지만, 각각의 child class가 하는 기능은 구별되어야 하기 때문입니다.

TODO 3-2: The update_current_path method of the BaseCommand class is slightly different from other methods. 
It has a @classmethod decorator and takes a cls argument instead of self. In Python, this is called a 
class method, and think about why it was implemented as a class method instead of a normal method.

Answer (You may write your answer in either Korean or English):
Class method는 특정 instance에 국한받지 않고 class 단위에서 자유롭게 사용할 수 있는 method이며, 상속 및 override가 가능합니다.
그래서 해당 Class의 변수에 대한 취급이 필요할 때 classmethod로 특정해 줄 필요가 있습니다.

"""
class BaseCommand:
    """
    Base class for all commands. Each command should inherit from this class and 
    override the execute() method.
    
    For example, the MoveCommand class overrides the execute() method to implement 
    the mv command.

    Attributes:
        current_path (str): The current path. Usefull for commands like ls, cd, etc.
    """

    current_path = os.getcwd()

    @classmethod
    def update_current_path(cls, new_path: str):
        """
        Update the current path.
        You need to understand how class methods work.

        Args:
            new_path (str): The new path. (Must be an relative path)
        """
        BaseCommand.current_path = os.path.join(BaseCommand.current_path, new_path)

    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize a new instance of BaseCommand.

        Args:
            options (List[str]): The command options (e.g. -v, -i, etc.)
            args (List[str]): The command arguments (e.g. file names, directory names, etc.)
        """
        self.options = options
        self.args = args
        self.description = 'Helpful description of the command'
        self.usage = 'Usage: command [options] [arguments]'

    def show_usage(self) -> None:
        """
        Show the command usage.
        """
        print(self.description)
        print(self.usage)

    def execute(self) -> None:
        """
        Execute the command. This method should be overridden by each subclass.
        """
        raise NotImplementedError