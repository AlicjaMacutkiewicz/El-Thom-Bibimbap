from rocketpy.sensors import ScalarSensor

class Thermometer(ScalarSensor):
    def measure(self, time, **kwargs):
        u = kwargs.get("u")
        env = kwargs.get("environment")
        altitude = u[2]
        true_temperature = env.temperature(altitude)
        self.measurement = true_temperature + self.constant_bias
        self.measured_data.append([time, self.measurement])
        return self.measurement

    def export_measured_data(self, filename, file_format="csv"):
        self._generic_export_measured_data(
            filename=filename, 
            file_format=file_format, 
            data_labels=("Time (s)", "Temperature (K)")
        )