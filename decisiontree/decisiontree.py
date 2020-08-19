import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings
import re


class DecisionTree:
    def __init__(self):
        self.tree = {}
        self.fail = {}
        self.ord_symbols = [r"<=", r">=", r"<", r">", "==", "!="]
        self.symbols = set(self.ord_symbols)

        self.root = None
        self.nodes = None

    def add_rule(self, name, conditions=None, value=None, next_step=None):
        if value and next_step:
            raise ValueError(f"You can select next step or outcome value")
        elif not value and not next_step:
            raise ValueError(f"Rule needs outcome, or next step")
        elif value:
            try:
                value = int(value)
            except ValueError:
                print(f"This is not number: {value} in {name} - {conditions}")
                return False

        if conditions:
            tmp_conditions = conditions
            conditions = conditions.replace(" ", "")
            pattern = "|".join(self.ord_symbols) + r"|\d+"
            conditions = re.findall(pattern, conditions)

            if len(conditions) == 1:
                try:
                    num = int(conditions[0])
                except ValueError as err:
                    print(f"This is not a number: {conditions[0]}")
                    num = None
                    return False
                ruld = self.get_ruld(eq=num)
                ruld.update({"next": next_step, "val": value})
                self._update_tree_rules(name, ruld)

            elif len(conditions) > 1:
                if not self.validate_parse(conditions):
                    print("Not valid")

                good_syms = self.symbols.copy()
                kwargs = {}
                for sym, num in zip(conditions[0::2], conditions[1::2]):
                    "This loop checks if there is not condition conflicts, and produces kwargs for rule dictionary"

                    if len(good_syms) <= 1:
                        print(f"To much rules or incorrect combination: {conditions}")
                        return None

                    if sym == '<' or sym == '<=':
                        if sym == "<=":
                            kw = "sme"
                        else:
                            kw = 'sm'
                        this_types = ["<", "<="]
                        incorect = ["=="]

                    elif sym == '>' or sym == '>=':
                        if sym == ">=":
                            kw = "bige"
                        else:
                            kw = 'big'
                        this_types = [">", ">="]
                        incorect = ["=="]

                    elif sym == '==' or sym == '!=':
                        if sym == "==":
                            kw = "eq"
                        else:
                            kw = 'diff'
                        this_types = ["==", "!="]
                        incorect = ["<", "<=", ">", ">="]
                    else:
                        raise ValueError(f"Symbol unknown: {sym}")

                    #                     print(good_syms)
                    for s in incorect:
                        try:
                            good_syms.remove(s)
                        except KeyError:
                            pass
                    try:
                        good_syms.remove(this_types[0])
                        good_syms.remove(this_types[1])
                    except KeyError as err:
                        if "!=" in good_syms:
                            pass
                        else:
                            print(f"Invalid rule: {name} - {conditions}")
                            print(f"{err}")
                            return False

                    kwargs.update({kw: num})
                ruld = self.get_ruld(**kwargs)
                if value:
                    ruld.update({"val": value})
                else:
                    ruld.update({"next": next_step})
                self._update_tree_rules(name, ruld)

            else:
                warnings.warn(f"Conditions are empty, using boolean condition: ''{tmp_conditions}'")
                self._add_bool(name, value, next_step)
        else:
            self._add_bool(name, value, next_step)

    def add_fail(self, name, value=None, next_step=None):
        if value and next_step:
            raise ValueError(f"Can not set value and next step as fail outcome for: {name}")
        elif name in self.fail.keys():
            warnings.warn(f"Overwriting fail outcome: {name}")
        self.fail.update({name: {'val': value, 'next': next_step}})

    def _add_bool(self, name, value, next_step):
        ruld = self.get_ruld(bl=True)
        if value:
            ruld.update({"val": value})
        else:
            ruld.update({"next": next_step})

        self._update_tree_rules(name, ruld)

    def get_ruld(self, bl=False, sm=False, sme=False, big=False, bige=False, eq=False, diff=False):
        """
        Smaller
        Smaller equal
        Bigger
        Bigger equal
        Equal
        Different
        """
        ruld = dict().fromkeys(['bl', 'sm', 'sme', 'big', 'bige', 'eq', 'diff', 'val', 'next'], False)
        if bl:
            ruld.update({'bl': True})

        elif sm and sme or big and bige or eq and diff:
            raise ValueError("Can not interpret conditions. Some types are not allowed simultaneously")

        elif sm or sme or big or bige or diff:
            if sm:
                sm = int(sm)
                ruld.update({'sm': sm})
            elif sme:
                sme = int(sme)
                ruld.update({'sm': sme})

            if big:
                big = int(big)
                ruld.update({'big': big})
            elif bige:
                bige = int(bige)
                ruld.update({'bige': bige})

            if diff:
                diff = int(diff)
                ruld.update({'diff': diff})

            max_val = sm or sme
            min_val = big or bige

            if max_val and min_val and min_val > max_val:
                raise ValueError(f"Conditions are impossible: {min_val} < {max_val}")
        elif eq:
            eq = int(eq)
            ruld.update({'eq': eq})

        return ruld

    def _update_tree_rules(self, name, new_rule):
        current = self.tree.get(name, [])
        current += [new_rule]
        self.tree.update({name: current})

    def build_tree(self):
        self.check_tree_keys()
        # self.check_condition_conflicts()
        valid, cmt, (root, nodes) = self.check_graph_sequence()

        if not valid:
            print(f"Not valid: {cmt}, there is cycle in tree")
        else:
            self.root = root
            self.nodes = nodes
            print(f"Tree is valid, build complete")

    def predict(self, **kwargs):
        # print(self.root)
        # print(self.nodes)
        next_name = self.root
        print(f"Kwargs: {kwargs}")
        while next_name:
            print()
            print(f"Checking: {next_name}")
            cur_name = next_name
            next_name = None
            outcome = False

            try:
                value = int(kwargs[cur_name])
            except KeyError:
                print(f"Can not predict, value is missing for: {cur_name}")
            except ValueError:
                print(f"This is not value: {kwargs[cur_name]}")
                return None

            cur_node = self.tree.get(cur_name)
            applies = False

            for rul in cur_node:
                next_name = rul['next']
                outcome = rul['val']
                print(rul)
                if rul['bl']:
                    if value:
                        applies = True
                    else:
                        continue
                elif rul['eq']:
                    if value == rul['eq']:
                        applies = True
                        print(f"equal")
                    else:
                        print(f"not equal")
                        continue
                else:
                    if rul['sm']:
                        v = rul['sm']
                        if value < v:
                            applies = True
                        else:
                            applies = False
                            continue
                    elif rul['sme']:
                        v = rul['sme']
                        if value <= v:
                            applies = True
                        else:
                            applies = False
                            continue
                    if rul['big']:
                        v = rul['big']
                        if value > v:
                            applies = True
                        else:
                            applies = False
                            continue
                    elif rul['bige']:
                        v = rul['bige']
                        if value >= v:
                            applies = True
                        else:
                            applies = False
                            continue
                    if rul['diff']:
                        v = rul['diff']
                        if value != v:
                            applies = True
                        else:
                            applies = False
                            continue
                    # or rul['sme'] or rul['big'] or rul['bige'] or rul['diff']

                if applies:
                    break

            if applies:
                print(f"Applies: {outcome}, {next_name}")
                if outcome:
                    return outcome
                elif next_name:
                    pass
            else:
                fail_cond = self.fail.get(cur_name)
                print(f"Not apply, {fail_cond}")
                try:
                    outcome, next_name = fail_cond['val'], fail_cond['next']
                except TypeError:
                    warnings.warn(f"Fail conditions does not exists, returning None, in {cur_name}")
                    return None
                if outcome is not None:
                    return outcome

    def check_tree_keys(self):
        fkeys = set(self.fail.keys())
        tree_keys = set(self.tree.keys())

        for key, rul_list in self.tree.items():
            if key not in fkeys:
                warnings.warn(f"{key} has not fail outcome. use 'dt.add_fail()'")
            for ruld in rul_list:
                nx = ruld['next']
                if nx and nx not in tree_keys:
                    raise ValueError(f"This next step does not exist: '{nx}' in '{key}' rules")

        for key, ruld in self.fail.items():
            nx = ruld['next']
            if nx and nx not in tree_keys:
                raise ValueError(f"This next step does not exist: '{nx}' in '{key}' rules")

    def _get_roots_with_nodes(self):
        nodes = {key: {'visited': 0, 'childs': set(), 'parents': set()}
                 for key in self.tree.keys()}
        roots = set(self.tree.keys())

        for name, rules in self.tree.items():
            "Removing non roots from roots, creating child, parent relations"
            for rul in rules:
                _next = rul.get('next', None)
                if _next:
                    this_node = nodes.get(name)
                    childs = this_node.get('childs')
                    childs.add(_next)
                    this_node.update({'childs': childs})
                    nodes.update({name: this_node})

                    child_node = nodes.get(_next)
                    parents = child_node.get('parents')
                    parents.add(name)
                    child_node.update({'parents': parents})
                    nodes.update({_next: child_node})

                    if _next in roots:
                        roots.remove(_next)

            fail_outcome = self.fail.get(name, None)
            if fail_outcome:
                _next = fail_outcome.get("next", None)
                if _next:
                    this_node = nodes.get(name)
                    childs = this_node.get('childs')
                    childs.add(_next)
                    this_node.update({'childs': childs})
                    nodes.update({name: this_node})

                    child_node = nodes.get(_next)
                    parents = child_node.get('parents')
                    parents.add(name)
                    child_node.update({'parents': parents})
                    nodes.update({_next: child_node})

                    if _next in roots:
                        roots.remove(_next)

                    if _next in roots:
                        roots.remove(_next)
        return roots, nodes

    def check_graph_sequence(self):
        roots, nodes = self._get_roots_with_nodes()

        if len(roots) != 1:
            err = f"Tree must have only one root: {roots}"
            print(err)
            return False, err

        root = list(roots)[0]
        visited = {root}
        goto_nodes = nodes.get(root).get('childs')

        n = 0
        while goto_nodes and n < 10:
            n += 1
            next_nodes = set()
            for cur_name in goto_nodes:
                if cur_name in visited:
                    err = f"This node was visited before: {cur_name}"
                    print(err)
                    return False, err

                checking_node = nodes[cur_name]
                vis_num = checking_node.get('visited', 0) + 1
                checking_node.update({'visited': vis_num})

                if 1 >= len(checking_node.get('parents')):
                    "Adding parents, its only 1, no conflicts"
                    visited.add(cur_name)
                    for ch in checking_node.get('childs'):
                        next_nodes.add(ch)
                elif vis_num >= len(checking_node.get('parents')):
                    "Joining paths"
                    visited.add(cur_name)
                    for ch in checking_node.get('childs'):
                        next_nodes.add(ch)
                else:
                    "Ignore / wait to join"

            goto_nodes = next_nodes

        if len(visited) == len(self.tree):
            return True, "Its ok to be ok", (root, nodes)
        else:
            not_vis = [node for node in nodes if node not in visited]
            return False, f"Some nodes was not visited {not_vis}", (None, None)

    def draw_graph(self):
        G = nx.Graph()
        # roots, nodes = self._get_roots_with_nodes()
        plt.figure(figsize=(16, 9))
        nx.draw(G)
        plt.show()

    def validate_parse(self, rules_to_check):
        size = len(rules_to_check)
        if size == 1:
            "Must be equal to value"
            try:
                int(rules_to_check[0])
                return True
            except ValueError as err:
                return False
        elif size == 0:
            print(f"pared rules are empty: {rules_to_check}")
            return False

        elif not size % 2:
            return True
        else:
            print("Rules are not even, len:", size, rules_to_check)


dt = DecisionTree()

dt.add_rule('want', "", next_step='worktime')
dt.add_fail('want', value=0)

dt.add_rule('worktime', "<1", value=5)
dt.add_rule('worktime', ">=1<10", next_step='criminal')
dt.add_rule('worktime', ">=10", next_step='member')

dt.add_rule('criminal', "", value=5)
dt.add_fail('criminal', next_step='member')

dt.add_rule('member', '<5', value=5)
dt.add_rule('member', '==8', value=15)
dt.add_rule('member', '>=5', value=35)
dt.add_fail('member', value=15)

dt.build_tree()

pred = dt.predict(want=True, worktime=10, criminal=False, member=9)
print(f"Predicted: {pred}")

# for key, val in dt.tree.items():
#     print(key, val)
#
# for key, val in dt.fail.items():
#     print(key, val)
