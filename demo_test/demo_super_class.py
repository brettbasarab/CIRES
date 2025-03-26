#!/usr/bin/env python

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

        self.statement = "I'm a rectangle!"

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

    def print_statement(self):
        print(self.statement)

    def this_method_defined_in_rectangle_class(self):
        print("This method is defined in the Rectangle class. This means it will be inherited by the Square class.")

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

        self.statement = "I'm a square!"

    def print_statement(self):
        print(self.statement)

    def this_method_defined_in_square_class(self):
        print("This method is defined in the Square class.")

# The Rectangle and Square classes each have a method called print_statement().
# The call to print_statement() in the square class will use that class's print_statement() method.
rect = Rectangle(7, 5)
square = Square(7)
rect.print_statement()
square.print_statement()

# The above is due to method resolution order, which can be seen by calling the __mro__ method:
print(Rectangle.__mro__)
print(Square.__mro__)

# Now, a method uniquely defined in the Rectangle class will be inhereted
# by the square class.
rect.this_method_defined_in_rectangle_class()
square.this_method_defined_in_rectangle_class()

# Contrast with a method defined only in the Square class.
square.this_method_defined_in_square_class()

