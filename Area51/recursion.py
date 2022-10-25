# Draw fibonacci using recursion and turtle graphics

import turtle


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


def drawFib(n):
    if n == 0:
        return
    else:
        drawFib(n - 1)
        turtle.forward(fib(n))
        turtle.left(90)
        turtle.forward(fib(n))


def main():
    drawFib(10)
    turtle.done()


if __name__ == "__main__":
    main()
