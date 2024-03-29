from .base_command import BaseCommand
import os
import shutil
from typing import List

class MoveCommand(BaseCommand):
    def __init__(self, options: List[str], args: List[str]) -> None:
        """
        Initialize the MoveCommand object.

        Args:
            options (List[str]): List of command options.
            args (List[str]): List of command arguments.
        """
        super().__init__(options, args)

        # Override the attributes inherited from BaseCommand
        self.description = 'Move a file or directory to another location'
        self.usage = 'Usage: mv [source] [destination]'

        # TODO 5-1: Initialize any additional attributes you may need.
        # Refer to list_command.py, grep_command.py to implement this.
        self.name = 'mv'
        self.options = options
        self.source = self.args[0] if self.args else ''
        self.destination = self.args[1] if self.args else ''
        self.interactive = '-i' in self.options
        self.verbose = '-v' in self.options
        
    def execute(self) -> None:
        """
        Execute the move command.
        Supported options:
            -i: Prompt the user before overwriting an existing file.
            -v: Enable verbose mode (print detailed information)
        
        TODO 5-2: Implement the functionality to move a file or directory to another location.
        You may need to handle exceptions and print relevant error messages.
        """
        
        if os.path.exists(self.destination):
            if not self.interactive:
                print(f"mv: cannot move '{self.source}' to '{self.destination}': Destination path '{self.destination}' already exists")
                return
            else:
                user_input = input(f"mv: overwrite '{self.destination}'? (y/n): ")
                if user_input.lower() != 'y':
                    print(f"mv: '{self.source}' not moved")
                    return

        try:
            shutil.move(self.source, self.destination)
            if self.verbose:
                print(f"mv: moving '{self.source}' to '{self.destination}'")
            if self.interactive and self.verbose:
                user_input = input(f"mv: overwrite '{self.destination}'? (y/n): ")
                if user_input.lower() != 'y':
                    print(f"mv: '{self.source}' not moved")
                    return
        except Exception as e:
            print(f"mv: error occurred - {e}")

        pass

    
    def file_exists(self, directory: str, file_name: str) -> bool:
        """
        Check if a file exists in a directory.
        Feel free to use this method in your execute() method.

        Args:
            directory (str): The directory to check.
            file_name (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = os.path.join(directory, file_name)
        return os.path.exists(file_path)
