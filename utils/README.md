TODO(): Finish classification loss and introductions to all methods, as well as finish UnitTests

# Basic Tools(utils/*)
## Activations
Utilize all activation functions, and we leave interfaces to assist anyone interested to add new activation functions.

It's easy to add new activation accomplishment in directory ``methods`` and then add the module into **activation_functions**. Since then, it's important to add pair **<lower string of your activation method, class_name>**  to dict **activators.py/ACTIVATION_MAP**.

Then we list some activation functions that we have accomplished.
### Activation Functions
1. Sigmoid
2. Tanh
3. ReLU
4. Leaky ReLU
5. Softmax

## Loss
As for loss, we use a similar design mode, but there is a little bit difference between them. We divide loss into two different kinds: **Regression Loss** and **Classification Loss**.

### Regression Loss
1. Mse
2. Mae
3. Huber

### Classification Loss

# Unittests
## Activations tests
test all activation functions, in order to guarantee all activation functions are right at least in unittest cases.

## Loss tests
