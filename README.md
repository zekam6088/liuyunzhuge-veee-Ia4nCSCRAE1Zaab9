## 快速幂

**快速幂（Fast Exponentiation）算法**解决这样一个问题：求解自然数的指数运算。计算 aba^bab 时，按照指数定义的朴素的方法是通过连续相乘：

ab=a×a×⋯×a⏟b次a^b = \underbrace{a \times a \times \cdots \times a}\_{b\text{次}}
ab=b次a×a×⋯×a​​

这种方法需要进行 b−1b-1b−1 次乘法，当 bbb 很大时（如 10910^9109），时间复杂度 O(b)O(b)O(b) 是完全不可接受的。

快速幂通过巧妙的二进制分解技术，将幂运算的时间复杂度从 O(b)O(b)O(b) 优化到 O(log⁡b)O(\log b)O(logb)。

考虑计算 a13a^{13}a13，将指数 13 用二进制表示：

13=11012=23+22+0+20=8+4+0+113 = 1101\_2 = 2^3 + 2^2 + 0 + 2^0 = 8 + 4 + 0 + 1
13=11012​=23+22+0+20=8+4+0+1

因此：

a13=a8+4+0+1=a8×a4×a0×a1a^{13} = a^{8 + 4 + 0 + 1} = a^8 \times a^4 \times a^0 \times a^1
a13=a8+4+0+1=a8×a4×a0×a1

而 a8=(a4)2=((a2)2)2 a^8 = (a^4) ^2 = ((a^2) ^2) ^2a8=(a4)2=((a2)2)2 ，分解后的幂次很容易计算

算法流程：

1. 初始化结果 1
2. 从最低位开始检查指数的二进制位
3. 如果当前位为 1，将当前的底数（a2xa^{2^x}a2x）乘入结果
4. 底数不断平方（不断计算 a0,a1,a2...a^0,a^1,a^2...a0,a1,a2...），指数右移一位
5. 重复直到指数的最高位 1 也被遍历

快速幂算法也可以从递归的角度来理解，这种理解方式更加直观。

ab={1if b=0(ab/2)2if b is evena×(a(b−1)/2)2if b is odda^b =
\begin{cases}
1 & \text{if } b = 0 \\
(a^{b/2})^2 & \text{if } b \text{ is even} \\
a \times (a^{(b-1)/2})^2 & \text{if } b \text{ is odd}
\end{cases}
ab=⎩⎨⎧​1(ab/2)2a×(a(b−1)/2)2​if b=0if b is evenif b is odd​

```
|  |  |
| --- | --- |
|  | long long quick_pow(long long base, long long exp) { |
|  | long long res = 1; |
|  | while (exp) { |
|  | if (exp & 1) { |
|  | res *= base; |
|  | } |
|  | base *= base; |
|  | exp >>= 1; |
|  | } |
|  | return res; |
|  | } |
```

### 带模数版本

更多的时候，我们要求解的是 ab mod ma^b \bmod mabmodm。这也可以用快速幂思想解决。快速幂模数版本的正确性基于模运算的分配律：

(a×b) mod m=[(a mod m)×(b mod m)] mod m(a \times b)\bmod m = [(a \bmod m) \times (b \bmod m)] \bmod m
(a×b)modm=[(amodm)×(bmodm)]modm

因此，我们可以直接对 a a a 取模，同时在算法每一步中，我们都对中间结果取模，这保证了最终结果的正确性，同时防止数值溢出。

不过我们**不能**直接对指数取模。指数 bbb 必须保持原值，因为：

ab mod m≠ab mod m mod ma^b \bmod m \neq a^{b \bmod m} \bmod m
abmodm=abmodmmodm

当然，既然复杂度是对数的，所以 bbb 大一些一般也无所谓。

```
|  |  |
| --- | --- |
|  | long long quick_pow(long long base, long long exp, long long mod) { |
|  | long long res = 1; |
|  | base %= mod;  // 先取模，防止初始base过大 |
|  | while (exp) { |
|  | if (exp & 1) { |
|  | res = (res * base) % mod; |
|  | } |
|  | base = (base * base) % mod; |
|  | exp >>= 1; |
|  | } |
|  | return res; |
|  | } |
```

不过，在某些特定情况下，我们可以使用**欧拉定理**来化简指数：

**欧拉定理**：如果 aaa 和 mmm 互质（即 gcd⁡(a,m)=1\gcd(a, m) = 1gcd(a,m)=1），那么：

aϕ(m)≡1(modm)a^{\phi(m)} \equiv 1 \pmod{m}
aϕ(m)≡1(modm)

其中 ϕ(m)\phi(m)ϕ(m) 是欧拉函数，表示小于 mmm 且与 mmm 互质的正整数的个数。

所以当 aaa 和 mmm 互质时，我们可以将指数对 ϕ(m)\phi(m)ϕ(m) 取模：

abmod  m=abmod  ϕ(m)mod  ma^b \mod m = a^{b \mod \phi(m)} \mod m
abmodm=abmodϕ(m)modm

这在某些数学和密码学应用中很有用，但**不是快速幂算法的必要部分**。代码略。

快速幂方法的时间复杂度是 O(log⁡b)O(\log b)O(logb)，循环次数等于指数的二进制位数，效率极高。

**演示**：计算 313mod  1003^{13} \mod 100313mod100

```
|  |  |
| --- | --- |
|  | 指数 13 = 1101(二进制) |
|  | 初始化: res = 1, base = 3 |
|  |  |
|  | 第1轮 (最低位为1): res = 1×3 = 3, base = 3² = 9 |
|  | 第2轮 (位为0):    res = 3,     base = 9² = 81 |
|  | 第3轮 (位为1):    res = 3×81 = 243 ≡ 43, base = 81² = 6561 ≡ 61 |
|  | 第4轮 (位为1):    res = 43×61 = 2623 ≡ 23 |
|  |  |
|  | 结果: 3¹³ ≡ 23 (mod 100) |
```

## 快速乘（防治溢出）

同样的思想也可以应用到乘法本身中。两个 32 位整数相乘，范围将达到 64 位；两个 64 位整数相乘，范围将达到 128 位。同样大小的数无法装入正确的结果。

快速乘（又称"龟速乘"）模仿快速幂的思想，将乘法运算转换为加法运算。核心思路是将 a×ba \times ba×b 看作是 bbb 个 aaa 相加，然后利用二进制分解来优化，这样就可以在中间结果下取模。

对于 a×ba \times ba×b，将 bbb 二进制分解：

b=∑i=0kbi⋅2i其中 bi∈{0,1}b = \sum\_{i=0}^{k} b\_i \cdot 2^i \quad \text{其中 } b\_i \in \{0,1\}
b=i=0∑k​bi​⋅2i其中 bi​∈{0,1}

那么：

a×b=a×∑i=0kbi⋅2i=∑i=0kbi⋅(a⋅2i)a \times b = a \times \sum\_{i=0}^{k} b\_i \cdot 2^i = \sum\_{i=0}^{k} b\_i \cdot (a \cdot 2^i)
a×b=a×i=0∑k​bi​⋅2i=i=0∑k​bi​⋅(a⋅2i)

```
|  |  |
| --- | --- |
|  | typedef long long ll; |
|  |  |
|  | // 快速乘：返回 (a * b) % mod，防止中间过程溢出 |
|  | ll quick_mul(ll a, ll b, ll mod) { |
|  | ll res = 0; |
|  | a %= mod; |
|  | while (b) { |
|  | if (b & 1) { |
|  | res = (res + a) % mod; |
|  | } |
|  | a = (a + a) % mod;  // a = 2a，相当于左移一位 |
|  | b >>= 1; |
|  | } |
|  | return res; |
|  | } |
|  |  |
|  | // 使用快速乘的快速幂 |
|  | ll quick_pow_safe(ll base, ll exp, ll mod) { |
|  | ll res = 1; |
|  | base %= mod; |
|  | while (exp) { |
|  | if (exp & 1) { |
|  | res = quick_mul(res, base, mod);  // 关键替换！ |
|  | } |
|  | base = quick_mul(base, base, mod);    // 关键替换！ |
|  | exp >>= 1; |
|  | } |
|  | return res; |
|  | } |
```

| 方法 | 时间复杂度 | 空间复杂度 | 防溢出能力 |
| --- | --- | --- | --- |
| 直接乘法 | O(1)O(1)O(1) | O(1)O(1)O(1) | 无 |
| 快速乘 | O(log⁡n)O(\log n)O(logn) | O(1)O(1)O(1) | 有 |

快速乘通过 O(log⁡n)O(\log n)O(logn) 次加法替代 O(1)O(1)O(1) 次乘法，实际上更慢了，所以也叫做“龟速乘”，这属于用时间换取了数值安全性。

## 浮点幂

如果是底数浮点，指数自然数，那么直接应用快速幂没有任何问题。但若指数是浮点数，这个问题会麻烦的多：浮点数指数无法直接进行二进制位操作，且误差会随着运算的拆分不断累积。

相比之下浮点幂的主要思想是利用自然对数变换法来计算浮点幂：

ab=eb⋅ln⁡(a)a^b = e^{b \cdot \ln(a)}
ab=eb⋅ln(a)

其中，自然对数和指数是常见且重要的函数，有快速且精确的办法来实现。

常见的库（如C++ 、Intel MKL、GNU Scientific Library）采用此类核心思路。

```
|  |  |
| --- | --- |
|  | // 伪代码示意 |
|  | if (a == 0.0) { |
|  | if (b > 0) return 0.0; |
|  | if (b == 0) return 1.0;  // 或 NaN，依标准而定 |
|  | return INFINITY;         // 或报错 |
|  | } |
|  | if (a == 1.0) return 1.0; |
|  | if (b == 0.0) return 1.0; |
|  | if (b == 1.0) return a; |
|  |  |
|  | result = exp(b * log(a)); # 对于一般情况 |
```

## 矩阵快速幂

已知矩阵 AAA，由于矩阵乘法满足结合律，指数为自然数时，仍可以利用快速幂思想求解 AnA^nAn。这最其深刻、最实用的扩展之一。它将快速幂的核心理念从标量运算成功迁移到了线性代数领域。

An={Iif n=0(An/2)2if n is evenA×(A(n−1)/2)2if n is oddA^n =
\begin{cases}
I & \text{if } n = 0 \\
(A^{n/2})^2 & \text{if } n \text{ is even} \\
A \times (A^{(n-1)/2})^2 & \text{if } n \text{ is odd}
\end{cases}An=⎩⎨⎧​I(An/2)2A×(A(n−1)/2)2​if n=0if n is evenif n is odd​

```
|  |  |
| --- | --- |
|  | typedef vectorlong long>> Matrix; |
|  |  |
|  | Matrix matrixMultiply(const Matrix& A, const Matrix& B, long long mod) { |
|  | int n = A.size(); |
|  | Matrix C(n, vector<long long>(n, 0)); |
|  | for (int i = 0; i < n; i++) { |
|  | for (int j = 0; j < n; j++) { |
|  | for (int k = 0; k < n; k++) { |
|  | C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod; |
|  | } |
|  | } |
|  | } |
|  | return C; |
|  | } |
|  |  |
|  | Matrix matrixPow(Matrix base, long long exp, long long mod) { |
|  | int n = base.size(); |
|  | // 初始化单位矩阵 |
|  | Matrix res(n, vector<long long>(n, 0)); |
|  | for (int i = 0; i < n; i++) { |
|  | res[i][i] = 1; |
|  | } |
|  |  |
|  | while (exp > 0) { |
|  | if (exp & 1) { |
|  | res = matrixMultiply(res, base, mod); |
|  | } |
|  | base = matrixMultiply(base, base, mod); |
|  | exp >>= 1; |
|  | } |
|  | return res; |
|  | } |
```

### 经典案例：斐波那契数列的矩阵解法

斐波那契数列的递推关系：

F(n)=F(n−1)+F(n−2)F(n) = F(n-1) + F(n-2)
F(n)=F(n−1)+F(n−2)

可以表示为矩阵形式：

[F(n)F(n−1)]=[1110]×[F(n−1)F(n−2)]\begin{bmatrix} F(n) \\ F(n-1) \end{bmatrix} =
\begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} \times
\begin{bmatrix} F(n-1) \\ F(n-2) \end{bmatrix}[F(n)F(n−1)​]=[11​10​]×[F(n−1)F(n−2)​]

递推得到：

[F(n)F(n−1)]=[1110]n−1×[F(1)F(0)]\begin{bmatrix} F(n) \\ F(n-1) \end{bmatrix} =
\begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}^{n-1} \times
\begin{bmatrix} F(1) \\ F(0) \end{bmatrix}[F(n)F(n−1)​]=[11​10​]n−1×[F(1)F(0)​]

由于我们可以快速计算矩阵的幂，我们就绕过了斐波那契数列的定义，使用对数次矩阵乘法的时间直接计算出了某一项。

```
|  |  |
| --- | --- |
|  | long long fibonacci_matrix(long long n, long long mod) { |
|  | if (n == 0) return 0; |
|  | if (n == 1) return 1; |
|  |  |
|  | Matrix base = {{1, 1}, {1, 0}}; |
|  | Matrix result = matrixPow(base, n - 1, mod); |
|  | return result[0][0];  // F(n) |
|  | } |
```

更一般的，对于 k 阶线性递推：

an=c1an−1+c2an−2+⋯+ckan−ka\_n = c\_1a\_{n-1} + c\_2a\_{n-2} + \cdots + c\_ka\_{n-k}
an​=c1​an−1​+c2​an−2​+⋯+ck​an−k​

构造转移矩阵：

M=[c1c2⋯ck−1ck10⋯0001⋯00⋮⋮⋱⋮⋮00⋯10]M = \begin{bmatrix}
c\_1 & c\_2 & \cdots & c\_{k-1} & c\_k \\
1 & 0 & \cdots & 0 & 0 \\
0 & 1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & 0
\end{bmatrix}M=​c1​10⋮0​c2​01⋮0​⋯⋯⋯⋱⋯​ck−1​00⋮1​ck​00⋮0​​

则：

[anan−1⋮an−k+1]=Mn−k+1×[ak−1ak−2⋮a0]\begin{bmatrix} a\_n \\ a\_{n-1} \\ \vdots \\ a\_{n-k+1} \end{bmatrix} =
M^{n-k+1} \times
\begin{bmatrix} a\_{k-1} \\ a\_{k-2} \\ \vdots \\ a\_0 \end{bmatrix}​an​an−1​⋮an−k+1​​​=Mn−k+1×​ak−1​ak−2​⋮a0​​​

本博客参考[wgetcloud](https://yiang.org)。转载请注明出处！
