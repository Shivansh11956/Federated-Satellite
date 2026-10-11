import flwr as fl
from model import create_model
from flwr.common import parameters_to_ndarrays

global_model = create_model()
best_loss = float("inf")

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        aggregated_result = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_result is not None:
            parameters, _ = aggregated_result
            weights = parameters_to_ndarrays(parameters)

            global_model.set_weights(weights)
            global_model.save("global_model.keras")

            print(f"Model saved after round {server_round}")

        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        global best_loss

        aggregated_result = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_result is not None:
            loss, _ = aggregated_result

            print(f"Round {server_round} Loss: {loss}")

            if loss < best_loss:
                best_loss = loss
                global_model.save("best_model.keras")
                print("Best model updated")

        return aggregated_result


strategy = SaveModelStrategy(
    min_fit_clients=2,
    min_available_clients=2,
    min_evaluate_clients=2,
    fraction_fit=1.0,
    fraction_evaluate=1.0
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy
)