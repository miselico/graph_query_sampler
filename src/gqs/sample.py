"""A sample specifies a portion of the queries"""


from typing import Callable, Union


class Sample:
    """
    A sample is a part of a split.

    You can specify which part of the data to be included and how many you want from this part.
    The final amounts will be sampled proportionally from the different matching data files.

    Parameters
        ----------
        selector : str
            The selector allows simple glob operations,
            for example `"/**/1qual/"` will select queries with one qualifier.
        amount : int, str, or a callable
            How many elements to sample from the sets indicated.
            This is specified by either:
                * a number indicating the amount
                * the string "*" indicating all
                * a callable that, given the total number available returns how many to sample.
            The latter can be used to control the amount depending on what is in the dataset,
               e.g., at most 10000  `lambda available: min(10000, available)` or
               10% of the data `lambda available: int(0.10 * available)`
        reify: bool, default=False
            Should the query be reified during the load?
            If True, triples which have qualifiers will be reified and an extra variable node will be added
            to represent the blank node. Then, for each qualifier a respective statement will be added with the
            respective node as the subject.In effect, the queries will have explicit qualifiers.

            This option conflicts with remove_qualifiers. Only one of these two can be True
        remove_qualifiers: bool, defaul = False
            Must qaulifiers be removed from the queries? If True, the qualifiers tensor will not contain anything.

            This option conflicts with reify. Only one of these two can be True

    """

    def __init__(self, selector: str, amount: Union[int, str, Callable[[int], int]], reify: bool = False, remove_qualifiers: bool = False) -> None:
        assert callable(amount) or (isinstance(amount, int) and amount >= 0) or amount == "*", \
            f"Illegal amount specification, got {amount}. Check the documentation."
        assert not (remove_qualifiers and reify), "Asked to both reify and remove qualifiers, this is almost certainly a mistake."
        self.selector = selector
        if isinstance(amount, int):
            int_amount: int = amount

            def exact(available: int) -> int:
                assert available >= int_amount, \
                    f"Not enough data available. Requested {int_amount}, but there is only {available}"
                return int_amount

            self._amount = exact
        elif amount == "*":
            self._amount = lambda available: available
        else:
            self._amount = amount  # type: ignore
        self.reify = reify
        self.remove_qualifiers = remove_qualifiers

    def amount(self, x: int) -> int:
        return self._amount(x)


def resolve_sample(
    choices: str,
) -> Sample:
    """Resolve a sample from a string representation."""
    split = choices.split(":")
    selector = split[0]
    amount: Union[str, int, Callable[[int], int]]
    if len(split) == 1:
        # there is no amount specification
        raise Exception("No amount specified")
    elif len(split) in {2, 3}:
        amount = split[1]
        if amount == "*":
            pass  # * is a correct psecification
        # elif amount.startswith("atmost"):
        #     amount = amount[6:]
        #     try:
        #         # using a different variable beacause of scoping of the nested fucntion.
        #         maximum_amount = int(amount)

        #     except ValueError as v:
        #         raise ValueError("Less than specification can only have an integer as its second part. E.g., *:atmost10000") from v

        #     def get_at_most(available: int) -> int:
        #         return min(available, maximum_amount)

        #     amount = get_at_most
        else:
            # convert to number
            try:
                amount = int(amount)
            except ValueError:
                try:
                    relative_amount = float(amount)
                except ValueError as v:
                    raise ValueError("Could not parse the amount part of the specification " + choices) from v

                def get_relative(absolute: int) -> int:
                    return int(relative_amount * absolute)

                amount = get_relative
    else:
        raise ValueError("sample specification contains more than two ':', specification was " + choices)

    reify = False
    remove_qualifiers = False
    if len(split) == 3:
        # there is an additional option
        option = split[2]
        if option == "reify":
            reify = True
        elif option == "remove_qualifiers":
            remove_qualifiers = True
        else:
            raise Exception(f"Invalid option specified. Expected one of reify|remove_qualifiers, got {option}")

    return Sample(selector, amount, reify=reify, remove_qualifiers=remove_qualifiers)
