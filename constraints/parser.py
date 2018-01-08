from rdflib import OWL, RDF, RDFS, ConjunctiveGraph, Namespace, URIRef
import logging

logging.basicConfig(level=logging.DEBUG)

class Node(object):
    def __init__(self, uri, parents):
        self.children = set()
        self.disjoints = set()
        self.uri = uri
        self.parents = set(parents)

    def __str__(self):
        return unicode(self.uri)

class ConstraintParser(object):
    def __init__(self, input_graphs):
        self.root = Node(OWL.Thing, [])
        self.g = ConjunctiveGraph()
        self.journal = {OWL.Thing : self.root}
        for path in input_graphs:
            print('Loading ', path)
            self.g.load(path, format='turtle')

    def get_node(self, uri):
        if uri in self.journal:
            return self.journal[uri]
        else:
            logging.debug('Creating node for : ' + unicode(uri))
            new_node = Node(uri, [])
            self.journal[uri] = new_node
            return new_node

    def get_all_children(self, uri):
        children = self.get_node(uri).children
        nodes_to_visit = children.copy()
        result = children.copy()
        childs_visited = 0
        while(len(nodes_to_visit) > 0):
            current_node = nodes_to_visit.pop()
            nodes_to_visit = nodes_to_visit.union(current_node.children)
            result.add(current_node)
            childs_visited += 1
            logging.debug("Child : " + str(childs_visited) + ' ' + unicode(current_node))
        return result

    def parse_hierarchy(self):
        types = self.g.objects(None, RDF.type)
        all_super_classes = set()
        # first find upper classes under root
        for t in types:
            super_classes = list(self.g.objects(t, RDFS.subClassOf))
            all_super_classes = all_super_classes.union(set(super_classes))
            logging.debug(unicode(t) + ' has super classes: ' + ''.join(super_classes))
            if len(super_classes) == 0 or (len(super_classes) == 1 and super_classes[0] == OWL.Thing):
                # super_class is root
                tmp_node = self.get_node(t)  # Node(t, [self.root])
                tmp_node.parents.add(self.root)
                self.root.children.add(tmp_node)
            else:
                tmp_node = self.get_node(t)
                tmp_node.parents.union(set([self.get_node(p) for p in super_classes]))
                for p in super_classes:
                    self.get_node(p).children.add(tmp_node)
        logging.debug(all_super_classes)

if __name__ == '__main__':
    yago_ns = Namespace('http://yago-knowledge.org/resource/')
    cp = ConstraintParser(['/home/nether-nova/Documents/data/yagoTypes10k.ttl',
                           '/home/nether-nova/Documents/data/yagoSimpleTaxonomy.ttl'])
    cp.parse_hierarchy()
    test_uri = yago_ns['wikicat_American_people']  # OWL.Thing

    print 'Give me all children of: ' + unicode(test_uri)

    direct_children = cp.get_node(test_uri).children
    print len(direct_children)
    print [unicode(c) for c in direct_children]
    all_children = cp.get_all_children(test_uri)
    print len(all_children)
    print [unicode(c) for c in all_children]