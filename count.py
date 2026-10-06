"""محاسبه دقیق تعداد پارامترهای آموزشی در شبکه‌های پرسپترون چندلایه (MLP)."""

from typing import List


def count_mlp_params(layers: List[int], bias: bool = True) -> int:
    """محاسبه پارامترهای لایه‌ها و نمایش جدول تفکیکی وزن و بایاس.

    ورودی:
        layers: لیست تعداد نرون‌ها [ورودی, لایه1, لایه2, ..., خروجی]
        bias: وضعیت فعال بودن بایاس در لایه‌ها (پیش‌فرض True)
    """
    total_weights = 0
    total_biases = 0

    print("=" * 60)
    print(
        f"{'لایه':<12} | {'ابعاد':<12} | {'وزن‌ها':<10} | {'بایاس‌ها':<8} | {'مجموع':<8}"
    )
    print("-" * 60)

    for i in range(len(layers) - 1):
        n_in = layers[i]
        n_out = layers[i + 1]

        weights = n_in * n_out
        biases = n_out if bias else 0
        layer_params = weights + biases

        total_weights += weights
        total_biases += biases

        dim_str = f"{n_in} -> {n_out}"
        layer_name = f"Layer {i + 1}"
        print(
            f"{layer_name:<12} | {dim_str:<12} | {weights:<10} | {biases:<8} | {layer_params:<8}"
        )

    grand_total = total_weights + total_biases
    model_size_kb = (grand_total * 4) / 1024
    adam_size_kb = (grand_total * 16) / 1024

    print("=" * 60)
    print(f"مجموع وزن‌ها:            {total_weights:,}")
    print(f"مجموع بایاس‌ها:           {total_biases:,}")
    print(f"کل پارامترهای آموزش‌پذیر:  {grand_total:,}")
    print("-" * 60)
    print(f"حجم مدل (FP32):           {model_size_kb:.2f} KB")
    print(f"حافظه مورد نیاز Adam:     {adam_size_kb:.2f} KB")
    print("=" * 60)

    return grand_total


if __name__ == "__main__":
    # مثال: ۵ ورودی، لایه پنهان اول ۱۰، لایه پنهان دوم ۸، خروجی ۳
    network_architecture = [5, 10, 8, 3]
    count_mlp_params(network_architecture, bias=True)
