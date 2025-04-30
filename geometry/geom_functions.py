import numpy as np

class SpatialGeometry:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Angular distance in radians using haversine formula."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        return 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def azimuth_angle(lat1, lon1, lat2, lon2):
        """Azimuth angle (bearing from lat1/lon1 to lat2/lon2) in radians."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.arctan2(x, y) + 2 * np.pi) % (2 * np.pi)
