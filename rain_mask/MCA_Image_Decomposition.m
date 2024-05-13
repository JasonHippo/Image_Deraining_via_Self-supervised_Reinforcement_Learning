function [Cartoon Texture Texture_Dict Cartoon_Dict] = MCA_Image_Decomposition(I, n, numIteration, DictSize, TextureDict, CartoonDict, Alg)
warning off;

K = DictSize;
[N1, N2] = size(I);
y = I;
Data = zeros(n^2, (N1 - n + 1) * (N2 - n + 1));
disp(size(Data));
cnt = 1; 
for j = 1:1:(N2-n+1)
    for i = 1:1:(N1-n+1)
        patch = y(i:i+n-1, j:j+n-1);
        Data(:,cnt) = patch(:); 
        cnt = cnt + 1;
    end;
end;

if isempty(TextureDict) && isempty(CartoonDict)
    % On-line learning
    Data = Data - repmat(mean(Data), [size(Data, 1) 1]);
    eps = 1e-10;
    denominator = sum(Data .^ 2) + eps;
    Data = Data ./ repmat(sqrt(sum(Data .^ 2)+eps), [size(Data, 1) 1]);
    param.K = DictSize;
    param.lambda = 0.15;
    param.iter = numIteration;
    Dictionary = mexTrainDL(Data, param);
    PHOG = zeros(81, K);
    VarTheta = zeros(1, K);
    for k=1:1:K
        atom = reshape(Dictionary(:, k), [n,n]);
        PHOG(:, k) = HOG(atom);
        [BW, thresh, gv, gh] = edge(atom, 'sobel');
        Theta = atan2(gv, gh);
        VarTheta(k) = var(Theta(:)); 
    end
    VarTheta = VarTheta / max(abs(VarTheta));
    AtomFeature = PHOG';
    
    % K-means clustering
    [IDX, C] = kmeans(AtomFeature, 2, 'maxiter', 100);
    Atom1 = find(IDX == 1)';
    Atom2 = find(IDX ~= 1)';
    VarTheta1 = mean(VarTheta(Atom1));
    VarTheta2 = mean(VarTheta(Atom2));
    
    if Alg == 2
        if norm(C(1, :)) <= norm(C(2, :))
            TextureAtoms = Atom1;
            CartoonAtoms = Atom2;
            %fprintf(1, 'Texture->Norm(C1) = %.2f, size(C1) = %d, VarTheta(C1) = %.2f.\n', norm(C(1, :)), size(Atom1, 2), VarTheta1);
            %fprintf(1, 'Cartoon->Norm(C2) = %.2f, size(C2) = %d, VarTheta(C2) = %.2f.\n', norm(C(2, :)), size(Atom2, 2), VarTheta2);            
        else
             TextureAtoms = Atom2;
            CartoonAtoms = Atom1;
            %fprintf(1, 'Texture->Norm(C2) = %.2f, size(C2) = %d, VarTheta(C2) = %.2f.\n', norm(C(2, :)), size(Atom2, 2), VarTheta2);
            %fprintf(1, 'Cartoon->Norm(C1) = %.2f, size(C1) = %d, VarTheta(C1) = %.2f.\n', norm(C(1, :)), size(Atom1, 2), VarTheta1);           
        end
    else
        if (VarTheta1 <= VarTheta2)
            TextureAtoms = Atom1;
            CartoonAtoms = Atom2;
            fprintf(1, 'Texture->Norm(C1) = %.2f, size(C1) = %d, VarTheta(C1) = %.2f.\n', norm(C(1, :)), size(Atom1, 2), VarTheta1);
            fprintf(1, 'Cartoon->Norm(C2) = %.2f, size(C2) = %d, VarTheta(C2) = %.2f.\n', norm(C(2, :)), size(Atom2, 2), VarTheta2);
        else
            TextureAtoms = Atom2;
            CartoonAtoms = Atom1;
            fprintf(1, 'Texture->Norm(C2) = %.2f, size(C2) = %d, VarTheta(C2) = %.2f.\n', norm(C(2, :)), size(Atom2, 2), VarTheta2);
            fprintf(1, 'Cartoon->Norm(C1) = %.2f, size(C1) = %d, VarTheta(C1) = %.2f.\n', norm(C(1, :)), size(Atom1, 2), VarTheta1);
        end
    end
    TextureDict = Dictionary(:, TextureAtoms);
    CartoonDict = Dictionary(:, CartoonAtoms);
    Dictionary = [TextureDict CartoonDict];
    Texture_Dict = TextureDict;
    Cartoon_Dict = CartoonDict;
else
    Dictionary = [TextureDict CartoonDict];
    Texture_Dict = TextureDict;
    Cartoon_Dict = CartoonDict;
end

Dictionary = Dictionary ./ repmat(sqrt(sum(Dictionary .^ 2)), [size(Dictionary, 1) 1]);
param.L = 10; 
param.eps = 0.1;
CoefMatrix = mexOMP(Data, Dictionary, param);
CoefMatrixTexture = CoefMatrix(1:size(TextureDict, 2), :);
CoefMatrixCartoon = CoefMatrix((size(TextureDict, 2) + 1):(size(TextureDict, 2) + size(CartoonDict, 2)), :);
Texture = RecoverImage(y, N1, N2, 0, Dictionary(:, 1:size(TextureDict, 2)), CoefMatrixTexture);
Cartoon = RecoverImage(y, N1, N2, 0, Dictionary(:, (size(TextureDict, 2) + 1):(size(TextureDict, 2) + size(CartoonDict, 2))), CoefMatrixCartoon);
return;

function [yout]=RecoverImage(y,N1, N2, lambda,D,CoefMatrix)
n=sqrt(size(D,1)); 
yout=zeros(N1,N2); 
Weight=zeros(N1,N2); 
i=1; j=1;
for k=1:1:(N1-n+1)*(N2-n+1),
    patch=reshape(D*CoefMatrix(:,k),[n,n]); 
    yout(i:i+n-1,j:j+n-1)=yout(i:i+n-1,j:j+n-1)+patch; 
    Weight(i:i+n-1,j:j+n-1)=Weight(i:i+n-1,j:j+n-1)+1; 
    if i<N1-n+1 
        i=i+1; 
    else
        i=1; j=j+1; 
    end;
end;

yout=(yout+lambda*y)./(Weight+lambda); 
return;