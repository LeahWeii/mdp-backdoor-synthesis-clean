# Reproducing Experimental Results

## Usage

Run the following command:

```bash
python gridworld_ex_finite_memory.py
```

## Parameters

- **`stoPar`** : $\alpha$ in the paper.

- **`p_obs`** :  $p_{obs}$ in the paper.

- **`sto_perturbed`**: The range of the values of $\alpha$ the attacker can use to perturb the system. 

- **`memory_length`**: The memory capacity of the trigger policy, specifying the number of past steps it can recall when making decisions.

- **`epsilon`** :  $\epsilon$ in the paper.

- **`episodes_num`**:  The number of episodes (iterations) over which the algorithm is executed.
