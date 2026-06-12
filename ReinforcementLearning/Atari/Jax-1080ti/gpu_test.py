#!/usr/bin/env python3
import sys
import jax
import jax.numpy as jnp

def test_gpu():
    print("=" * 50)
    print("Checking JAX devices...")
    devices = jax.devices()
    print(f"Available devices: {devices}")

    # Check if GPU is detected
    gpu_found = False
    for device in devices:
        if device.platform == 'gpu':
            gpu_found = True
            print(f"Successfully found GPU: {device.device_kind}")
            break

    if not gpu_found:
        print("WARNING: No GPU detected! JAX is running on CPU.")

    print("\nRunning a simple matrix multiplication on JAX...")
    # Generate some random matrices
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jax.random.normal(key, (1000, 1000))

    # Matrix multiplication
    z = jnp.matmul(x, y)
    # Block until the multiplication is computed (JAX uses asynchronous dispatch)
    z.block_until_ready()
    print("Matrix multiplication completed successfully.")

    print("\nRunning a simple differentiation (grad) test...")
    # Define a simple function: f(x) = x^3 + 2x^2 + 5x
    def f(x):
        return x**3 + 2*x**2 + 5*x

    # Take the derivative of f
    df = jax.grad(f)

    # Calculate df(3.0) -> expected: 3*(3.0^2) + 4*(3.0) + 5 = 27 + 12 + 5 = 44.0
    val = 3.0
    df_val = df(val)
    print(f"Function: f(x) = x^3 + 2x^2 + 5x")
    print(f"Analytical derivative at x = {val}: f'(x) = 3x^2 + 4x + 5")
    print(f"JAX gradient result: f'({val}) = {df_val}")

    assert jnp.isclose(df_val, 44.0), f"Grad test failed, expected 44.0 but got {df_val}"
    print("Differentiation test completed successfully.")

    print("=" * 50)
    if gpu_found:
        print("JAX GPU test passed successfully!")
    else:
        print("JAX CPU test passed (but no GPU was detected)!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_gpu()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        sys.exit(1)
