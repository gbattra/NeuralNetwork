# unwraps Theta array into separate Theta matricies


def unwrap_theta(W, L_in, hidden_layer, L_out):

    # unwraps Thetas into initial thetas
    Theta_1_size = hidden_layer * (L_in + 1)
    Theta_1_flat = W[0:Theta_1_size]
    Theta_2_flat = W[Theta_1_size:]

    # reshape flat thetas into correct dimensions
    Theta_1 = Theta_1_flat.reshape(hidden_layer, L_in + 1)
    Theta_2 = Theta_2_flat.reshape(L_out, hidden_layer + 1)

    return Theta_1, Theta_2
