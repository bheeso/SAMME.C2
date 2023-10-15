# SAMME.C2
This project provides the source code for the SAMME.C2 algorithm, a hybrid approach that combines the strengths of two distinct algorithms:
- SAMME: This algorithm is an AdaBoost variant designed for multi-class classification.
- Cost-sensitive learning, as seen in Ada.C2.
  
SAMME.C2 leverages the power of both SAMME and cost-sensitive learning to enhance its effectiveness in various classification tasks.

# Algorithm

\begin{algorithm}[H]
\LinesNumberedHidden
\SetAlgoNoLine
	\KwData{$\ \bm{x}_i \in X$, $y_i \in Y = \{1,2,\ldots,K\}$}
	\KwIn{$C(y_i) \in (0,1]$, $T$}
	\KwOut{Final classifier $\ H(\bm{x}_i)= \underset{k}{\mathrm{argmax}} {\  \sum_{t=1}^{T} \alpha_t I(h_t(\bm{x}_i) = k)}$}
	Set initial sample weights to be equally distributed:  $D_1(i) = \frac{1}{N}, \quad i =1,2,\ldots,N$ \;
	
	\For{$t=1, \ldots, T$}{
		Train weak classifier using distribution $D_t$\;
		
		Obtain weak classifier $h_t: X \rightarrow k \in \{1,2,\ldots,K\}$\; 
		
		Compute error rate  $\epsilon_t = \dfrac{\sum_{i=1}^{N} D_t(i) I(y_i \ne h_t(\bm{x}_i))}{\sum_{i=1}^N D_t(i)}$ \;
		
		Calculate weight $\alpha_t = \log\!\Big(\dfrac{1-\epsilon_t}{\epsilon_t}\Big) + \log(K-1)$ \;
		
		Update sample weights $D_{t+1}(i) = \dfrac{C(y_i) \, D_t(i) \exp(-\alpha_t I(y_i = h_t(\bm{x}_i)))}{\sum_{j=1}^{N} C(y_j) \, D_t(j) \exp(-\alpha_t I(y_j = h_t(\bm{x}_j)))}$ \;
	}
	\caption{\texttt{SAMME.C2}: Cost-sensitive multi-class AdaBoost}\label{alg:sammec2}
\end{algorithm}

# Contributing 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# Reference
So B, Boucher JP, Valdez EA (2021) Cost-sensitive multi-class adaboost for understanding driving behavior based on telematics. ASTIN Bulletin: The Journal of the IAA 51(3):719–751
