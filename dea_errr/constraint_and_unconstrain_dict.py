def constrain_dict(dictionary):
    # Find the minimum and maximum values for the keys and values
    all_coords = []
    for key, tuple_list in dictionary.items():
        all_coords.append(key[0])
        all_coords.append(key[1])
        for original_tuple in tuple_list:
            all_coords.append(original_tuple[0])
            all_coords.append(original_tuple[1])
    minimum = min(all_coords)
    maximum = max(all_coords)

    # Constrain the keys and values using the minimum and maximum values
    constrained_dict = {}
    for key, tuple_list in dictionary.items():
        constrained_key = ((key[0] - minimum) / (maximum - minimum), (key[1] - minimum) / (maximum - minimum))
        constrained_tuple_list = []
        for original_tuple in tuple_list:
            constrained_x = (original_tuple[0] - minimum) / (maximum - minimum)
            constrained_y = (original_tuple[1] - minimum) / (maximum - minimum)
            constrained_tuple_list.append((constrained_x, constrained_y))
        constrained_dict[constrained_key] = constrained_tuple_list
    return (constrained_dict,minimum,maximum)


def unconstrain_dict(constrained_dict, minimum, maximum):
    original_dict = {}
    for constrained_key, constrained_tuple_list in constrained_dict.items():
        original_key_x = constrained_key[0] * (maximum - minimum) + minimum
        original_key_y = constrained_key[1] * (maximum - minimum) + minimum
        original_key = (original_key_x, original_key_y)
        original_tuple_list = []
        for constrained_tuple in constrained_tuple_list:
            original_x = constrained_tuple[0] * (maximum - minimum) + minimum
            original_y = constrained_tuple[1] * (maximum - minimum) + minimum
            original_tuple_list.append((original_x, original_y))
        original_dict[original_key] = original_tuple_list
    return original_dict

