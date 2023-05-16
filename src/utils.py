class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            elif isinstance(value, list):
                setattr(self, key, [Config(**item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)
    def __str__(self):
        result = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result.append(f"{key}: {str(value)}")
            elif isinstance(value, list):
                result.append(f"{key}: {[str(item) for item in value]}")
            else:
                result.append(f"{key}: {value}")
        result = [item.replace('"', '').replace("'", "") for item in result]
        result = "{ " + ", ".join(result) + " }"
        return result

class MetricAvg:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.reset_all()
    
    def reset_all(self):
        self.metrics = {
            metric_name: [0, 0]
            for metric_name in self.metric_names
        }

    def reset(self, metric_name):
        self.metrics[metric_name] = [0, 0]
    
    def update(self, metric_name, value, cnt):
        self.metrics[metric_name][0] += value
        self.metrics[metric_name][1] += cnt
    
    def get(self, metric_name):
        if self.metrics[metric_name][1] == 0:
            return None
        return self.metrics[metric_name][0] / self.metrics[metric_name][1]
    
    def get_all(self):
        return {
            metric_name: self.get(metric_name)
            for metric_name in self.metrics.metric_names
        }