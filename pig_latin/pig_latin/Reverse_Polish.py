import operator
import sys


# number storage declaration
stack = []

# operator dictionary
opdict = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    'x' : operator.mul,
    '/' : operator.truediv,
    '%' : operator.mod
}

def entrymenu():
    # entry dialog
    print("Enter a number OR operator and press return:")
    print("Enter h for help, commands, and reverse polish notation")

def helpmenu():
    # some explanation of reverse polish, and more options
    print("Reverse polish calculators work by entering numbers, and then the operator subsequently.")
    print("Operators operate on the last two numbers that have been entered into the calculator.")
    print("Ex: 5 [enter] 5 [enter] x [enter] = (5x5) = 25\n")
    print("C:          Clear operations")
    print("exit:       Exit calculator\n")
    print("Press any key to return to the calculator")
    x = input()
    # for the off-chance that a user wants to exit the program from the help menu
    if x == 'exit':
        sys.exit()
    entrymenu()

entrymenu()

while 1 == 1:
    x = input()
    # non-calculating commands
    if x.lower() == 'h' or x.lower() == 'help':
        helpmenu()
    elif x.lower() == 'c':
        stack = []
        print(0)
    elif x == 'exit':
        sys.exit()

    # calculating operations
    else:
        try:
            # add the number to the list if it is able to be converted to an int
            stack.append(int(x))

        except(ValueError):
            # perform the operation using the operator dict if the entry is not a number
            if x == "":
                print("Invalid entry. Try again.\n")
                entrymenu()
                print(stack[-1])
                continue
            # error handling for case for too few numbers
            if len(stack) < 2:
                print("There are too few numbers to calculate with. Enter more numbers.\n")
                entrymenu()
                # stack is at zero if there are too few numbers and a non-number is entered, but not if only a number is entered
                if stack != []:
                    print(stack[-1])
                else:
                    print(0)
                continue
            # correctly using polish notation
            if stack != []:
                try:
                    operation_result = opdict[str(x)](stack[-2], stack[-1])
                    print(operation_result)
                    stack.remove(stack[-1])
                    stack.remove(stack[-1])
                    stack.append(operation_result)
                # dividing by zero error handling. Resets the number stack
                except(ZeroDivisionError):
                    print("Can't divide by zero. Operations reset.\n0")
                    stack = []
                    continue
                # error handling for a non-operator, non-number entry
                except(KeyError):
                    print("Invalid entry. Try again.\n")
                    entrymenu()
                    continue
