class Order:
    def __init__(self, origin, destination, dropoff_date, target_delivery_date):
        self.origin = origin
        self.destination = destination
        self.dropoff_date = dropoff_date
        self.target_delivery_date = target_delivery_date
        self.assigned_voyage = None

class Voyage:
    def __init__(self, origin, destination, departure_date, arrival_date, capacity):
        self.origin = origin
        self.destination = destination
        self.departure_date = departure_date
        self.arrival_date = arrival_date
        self.capacity = capacity
        self.booked_capacity = 0
        self.booked_orders = []

def book_order(order, voyages):
    # Check if there are any voyages that match the origin and destination of the order
    matching_voyages = [voyage for voyage in voyages if voyage.origin == order.origin and voyage.destination == order.destination]
    
    # If there are no matching voyages, return None
    if not matching_voyages:
        return None
    
    # Find the voyage with earliest departure date that satisfies the order's needs and has available capacity
    sorted_voyages = sorted(matching_voyages, key=lambda voyage: voyage.departure_date)
 
    for voyage in sorted_voyages:
        if (voyage.departure_date >= order.dropoff_date and
            voyage.arrival_date <= order.target_delivery_date and
            voyage.booked_capacity < voyage.capacity):
            voyage.booked_capacity += 1
            order.assigned_voyage = voyage
            return True
    return False

# Why flexport
# Important field - visibility and manageability into freight, covid
# Self-high ownership culture, digital, data-driven first approach, past experience, innovation, supportive nature, emphasis on engineering excellence - coding standard,
# Data, api, customization - insurance, data platform