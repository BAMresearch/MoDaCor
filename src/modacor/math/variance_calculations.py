def add(x, vx, y, vy=0):
    """
    Return the sum of x and y, along with the propagated variance.

    Parameters:
        x (float): First value.
        vx (float): Variance of x.
        y (float): Second value.
        vy (float, optional): Variance of y. Defaults to 0.

    Returns:
        tuple: (x + y, vx + vy)
    """
    result = x + y
    variance = vx + vy
    return result, variance


def subtract(x, vx, y, vy=0):
    """
    Return the difference (x - y), along with the propagated variance.

    Parameters:
        x (float): Minuend.
        vx (float): Variance of x.
        y (float): Subtrahend.
        vy (float, optional): Variance of y. Defaults to 0.

    Returns:
        tuple: (x - y, vx + vy)
    """
    result = x - y
    variance = vx + vy
    return result, variance


def multiply(x, vx, y, vy=0):
    """
    Return the product of x and y, with propagated variance using first-order error propagation.

    Parameters:
        x (float): First value.
        vx (float): Variance of x.
        y (float): Second value.
        vy (float, optional): Variance of y. Defaults to 0.

    Returns:
        tuple: (x * y, y^2 * vx + x^2 * vy)
    """
    result = x * y
    dx_, dy_ = y, x
    variance = dx_**2 * vx + dy_**2 * vy
    return result, variance


def divide(x, vx, y, vy=0):
    """
    Return the quotient (x / y), with propagated variance using first-order error propagation.

    Parameters:
        x (float): Numerator.
        vx (float): Variance of x.
        y (float): Denominator.
        vy (float, optional): Variance of y. Defaults to 0.

    Returns:
        tuple: (x / y, (∂/∂x)^2 * vx + (∂/∂y)^2 * vy)
                where partial derivatives are:
                    ∂/∂x = 1 / y
                    ∂/∂y = -x / y^2
    """
    result = x / y
    dx_, dy_ = 1 / y, -x / (y**2)
    variance = dx_**2 * vx + dy_**2 * vy
    return result, variance
