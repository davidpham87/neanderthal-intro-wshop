# Introduction to Linear Algebra

# Strategy

 Going from example first, then abstract.

# Goals

 - Introduce heuristically basic concepts of linear algebra to neophytes with
   Clojure knowledge.

Caveat: I will consciously ignore a few mathematical details, to emphasize on
the intuition [this hurts me as a mathematician].

# What is linear algebra?

Crudely speaking and for a software engineer, linear algebra is the science and
art of adding and multiplying numbers really fast without user defined loop.

Mathematically, linear algebra studies vector spaces and linear transformation
between spaces.

# Why caring about linear algebra?

These are example of area where linear algebra is applied the most are:

- Statistics: linear algebra provides a vocabulary and
  tools to solve problem in high dimensions.
- Machine Learning: linear algebra is abused for computer vision, natural
  language processing, speech recognition, and gaming agent.
- Simulations of [stochastic] systems (Markov chains): when studying systems
  where variable can influence each other, linear algebra might help to find
  the equilibrium state.
- Study of graphs/networks: graphs and networks can be represented as a
  matrix where each row represent the outward connection from a vertex to
  another vertex.
- Other ares: solving partial differential equations, optimization and
  budgeting, computer graphics [more details later].

# Why learning a specialized Linear Algebra Library?

It is true that one could heuristically define most linear algebra operations
as such

``` clojure
(def plus #(apply mapv + %&))
(defn times [alpha x] (mapv (partial * alpha) x))
(defn dot [x y] (reduce + (mapv * x y)))
;; Suppose matrix are row majors
(defn transpose [Y] (apply mapv vector Y))
(defn %*% [X Y]
  (mapv (fn [x] (mapv #(dot x %) (transpose Y))) X))

(plus [1 2] [0 1])
(times 3 [1 2])
(dot [1 2] [0 1])

(%*% [[1 0] [0 1]]
     [[1 2] [3 4]])

(%*% [[0 1] [1 0]]
     [[1 2] [3 4]])

(%*% [[-1 0] [0 1]]
     [[1 2] [3 4]])
```

## Speed and Algorithms

The reason linear algebra is a necessity in practice for these fields is thanks
to the concept and the hardware implementation of *single instruction multiple
data* (`SIMD`), in other words the art of applying the same operation to
multiple data point. Basically, CPU and GPU have specialized instructions to
perform linear algebra operations which speeds up operations by magnitude of
orders (depending on the size of the problem).

The gain speed allowed to create novel algorithms (such as the bootstrap
algorithm in statistics).

Depending on the structure of your problem, you could leverage the shape of
your matrices to speed up even more the computations.


# Linear Algebra Concepts

## Vector Spaces

The most used vector space is probably the cartesian product of $\mathbb{R}$,
that is $\mathbb{R} \times \mathbb{R} = \mathbb{R}^2$, and generally
$\mathbb{R}^n$.

Other vector spaces: the space of function from $\mathbb{R}$ to
$\mathbb{R}$.

## Subspace and basis

## Linear Mappings

Another example of mapping: the derivative of a function in the space of
function that can be derived is a linear operation, as

$T(\alpha f+ \beta g) = (\alpha f + \beta g)' = \alpha f' + \beta  g' = \alpha T(f) + \beta T(g)$.


- What is Linear Algebra? Start with example of geometry [vector/planes], then
  speak about linear transformation as mappings/functions [speak about
  projection, bijectivity], Transition to matrix showing a one to one
  relationship between function to matrices. Make the case of composition of
  mapping is multiplication of matrices?

- Start making example with scaling and rotation, extend to translate.

- Linear Algebra in Clojure: speak about Neanderthal API and link to numpy?
