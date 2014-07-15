load pos
load neg
load positiveGrad
load negativeGrad

fprintf('maximal distance of positiveGrad:  %f\n', 10000000*max(abs(pos-positiveGrad)));
fprintf('maximal distance of negativeGrad:  %f\n', 10000000*max(abs(neg-negativeGrad)));