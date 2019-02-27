y=[X1 X2 X3] * [h1 h2 h3]'


X = [X1 X2 X3]
h = inv(X'*X)*X'*y


newX1 = project([poly])*X1

newy = project([poly])*y

h = newX1\newy

m = (I-X*(inv(X'*X)*X')) * y

y - X*(inv(X'*X)*X')*y


MOST IMPORTANT TO GET DOWN
1. input specification!!!  (decide what functionality to hack away)
2. makeimagestack
3. getcanonicalhrf.m
4. constructstimulusmatrices.m
5. constructpolynomialmatrix.m
6. PCA details...
