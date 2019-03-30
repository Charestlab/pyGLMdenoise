from glmdenoise.report import Report


class Output(object):

    def __init__(self):
        pass

    def determine_location(self, sample_file):
        pass

    def determine_location_in_bids(self, bids, sub, ses, task):
        pass

    def create_report(self):
        return Report()

    def save_image(self, imageArray, name):
        pass

    def save_variable(self, var, name):
        pass
