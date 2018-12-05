function [X, train_error, test_error, run_time, obj_val] = OOE(X, train_triplet, test_triplet, d, lambda, mu, gamma0, p, NITER)


	if ~exist('lambda', 'var') || isempty(lambda)
		lambda = 0;
	end
	if ~exist('p', 'var') || isempty(p)
		p = 1;
	end
	if ~exist('gamma0', 'var') || isempty(gamma0)
		gamma0 = 10;
	end
	if ~exist('mu', 'var') || isempty(mu)
		mu = 0.1;
	end

	% NITER = 100;

	rho = 1.05;
	
	% Determine number of objects
	n = max(train_triplet(:));
	train_triplet(any(train_triplet == -1, 2), :) = [];
	no_train = size(train_triplet, 1);
	no_test = size(test_triplet, 1);

	% Initialize some variables
	% X: d*n
	% X = randn(d, n);
	G = X' * X;
	G1 = G;
	G2 = G;
	z1 = zeros(no_train, 1);
	z2 = zeros(no_train, 1);
	Z3 = zeros(n, n);
	Z4 = zeros(n, n);
	run_time = zeros(NITER+1, 1);
	train_error = zeros(NITER+1, 1);
	test_error = zeros(NITER+1, 1);
	obj_val = zeros(NITER+1, 1);


	% Initialization of K
	for q = 1:no_train
		tmp = zeros(n, n);
		i = train_triplet(q, 1);
		j = train_triplet(q, 2);
		k = train_triplet(q, 3);
		tmp(i, j) = -1;
		tmp(j, i) = -1;
		tmp(j, j) =  1;
		tmp(k, k) = -1;
		tmp(i, k) =  1;
		tmp(k, i) =  1;
		K(q, :) = tmp(:); 
	end	
    
	inver_K = eye(n^2)-K'*inv(eye(no_train)+K*K')*K;

	% inver_K = eye(no_train)-K'*inv(eye(no_train)+K*K')\K;

	% convergence tolerance
	tol = 1e-5;
	iter = 1;
	converged = false;

	Eq = gamma0*ones(q, 1)-K*G(:);
	E1 = Eq;
	E2 = -Eq;

	while ~converged
		tt = clock;
		% Update E1 & E2
		update_E;

		% Update G1 & G2
		[G1, nuclear_norm] = proximal_nuclear(G+Z3./mu, lambda/mu);

		G2 = proximal_sdp(G+Z4./mu);	

		% Update G
		C1 = (E2 - E1) + (z2-z1)/mu + 2 * gamma0;
		C2 = G1-Z3./mu+G2-Z4./mu;
		G = update_G(K, inver_K, C1, C2, n);
		Eq = gamma0-K*G(:);

		% Update multiplier
		update_multiplier;

		objective_cal;
        
        mu = min(1e5, mu*rho);

		run_time(iter) = run_time(iter) + etime(clock, tt);

		% Evaluate Error
		error_eval;

 		% if ((iter > 2) && (obj_val(iter-1)-obj_val(iter)<.00005)) || (iter > NITER)
 			% converged = true;
 		% end

		iter = iter+1;

        if iter > NITER
            converged = true;
        end
	end

	idx = find(train_error ~= 0);
	train_error = train_error(idx(end));
	idx = find(test_error ~= 0);
	test_error = test_error(idx(end));
	idx = find(obj_val ~= 0);
	obj_val = obj_val(idx(end));

	function  update_E
		S1 =  Eq - z1./mu;
		S2 = -Eq - z2./mu;
		Omega1 = double(S1>0);
		OmegaS1 = 1-Omega1;
		Omega2 = double(S2>0);
		OmegaS2 = 1-Omega2;
		if p == 1 % L1 
			Es1 = max(S1 - 1/mu, 0);
			Es1 = Es1+min(S1 + 1/mu, 0);
			E1 = Omega1.*Es1 + OmegaS1.*S1;
			Es2 = max(S2 - 1/mu, 0);
			Es2 = Es2+min(S2 + 1/mu, 0);
			E2 = Omega2.*Es2 + OmegaS2.*S2;
		elseif p == 2 % L2
			E1 = Omega1.*S1/(1+2*1/mu)+OmegaS1.*S1;
			E2 = Omega2.*S2/(1+2*1/mu)+OmegaS2.*S2;
		else
			errmsg('p can be 1 or 2');
		end	
	end

	function [Y, s] = proximal_nuclear(X, cons)
		[U, S, V] = svd(X, 'econ');
		S = max(S-cons, 0);
		Y =  U * S * V';
		s = sum(diag(S));
	end

	function Y = proximal_sdp(X)
		B = 1/2*(X+X');
		% [V, L] = eig(B);
		% H = V*(abs(L))*V';
		[~, S, V] = svd(B, 'econ');
		H = V * S * V';
		Y = 1/2*(B+H);
	end

	function G = update_G(K, inver_K, C1, C2, n)
		g = (1/2) * inver_K * (K' * C1 + C2(:) ) ;
		G = reshape(g, [n n]);
	end

	function update_multiplier
		z1 = z1+mu*(E1-Eq);
		z2 = z2+mu*(E2+Eq);
		Z3 = Z3+mu*(G-G1);
		Z4 = Z4+mu*(G-G2);
	end

	function objective_cal
		if p == 1
			hinge1 = sum(max(-E1, 0));
			hinge2 = sum(max(-E2, 0));
		else
			hinge1 = sum(max(-E1, 0).^2);
			hinge2 = sum(max(-E2, 0).^2);
		end

		obj_val(iter) = hinge1 + hinge2 + lambda*nuclear_norm;
	end

	% function inner_val = inner_product(A, B)
	% 	if size(A, 2) == 1 || size(A, 1) == 1
	% 		inner_val = sum(A.*B);
	% 	else
	% 		inner_val = sum(sum(A.*B));
	% 	end
	% end

	% function Phi_val = Phi_cal(A, B, multiplier, mu)
	% 	Phi_val = inner_product(multiplier, A - B) + 0.5*mu*(sumsqr(A - B));
	% end

	function error_eval
		D = bsxfun(@plus, bsxfun(@plus, -2 .* G, diag(G)), diag(G)');
		no_train_viol = sum(D(sub2ind([n n], train_triplet(:, 1), train_triplet(:, 2))) > ...
						D(sub2ind([n n], train_triplet(:, 1), train_triplet(:, 3))));
		 no_test_viol = sum(D(sub2ind([n n], test_triplet(:, 1), test_triplet(:, 2))) > ...
						D(sub2ind([n n], test_triplet(:, 1), test_triplet(:, 3))));
		train_error(iter) = no_train_viol ./ no_train;
		 test_error(iter) =  no_test_viol ./ no_test;
	end
end