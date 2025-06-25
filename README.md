# Two-step regularization for an inverse problem of the 1+1 dimensional wave equation
Tools for numerical inversion of the wave equation using a two-step inversion method.
First, the Iterative Time Reversal Control method is used to calculate travel time volumes,
after which Total Variation regularization is used to denoise the derivatives needed to compute
the wave speed. Both steps are implemented by the class `ITRC` in `src/ITRC.py`.

Structure of the project:
- analysis: contains Jupyter notebooks that showcase the behavior of the `ITRC` class and its
associated computational methods.
- src: contains the source code used in the analyses. In particular, `src/ITRC.py` contains the main
logic of the inversion method, while `src/simulation.py` contains methods to simulate the forward problem.
Regularized differentiation is implemented in `reg_diff.py`: this uses gradient descent code from `tv_gdbb.py`.

## MIT Licence

Copyright © 2025 Daniel Repo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.